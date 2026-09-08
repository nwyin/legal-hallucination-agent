import subprocess
import os
import concurrent.futures
import psutil
import pwd
import json
from typing import List, Optional

from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm

# Import SearchResult from open_search module
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'open_search'))
try:
    from search import SearchResult  # type: ignore
except ImportError:
    # Fallback if import fails
    class SearchResult:  # type: ignore
        def __init__(self, title, url, snippet, source, published_date=None, metadata=None, result_id=None):
            self.title = title
            self.url = url
            self.snippet = snippet
            self.source = source
            self.published_date_raw = published_date
            self.published_date = None
            self.metadata = metadata or {}
            self.result_id = result_id

from .metadata_filters import (
    MetadataFilterPolicy,
    apply_metadata_filters,
    compute_prefetch_k,
)

# Get SLURM CPU allocation
SLURM_CPUS_ON_NODE = int(os.environ.get("SLURM_CPUS_ON_NODE", "1"))
SLURM_CPUS_PER_TASK = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

# Total available CPUs (threads) - use the smaller of the two if both are set
TOTAL_CPUS = min(SLURM_CPUS_ON_NODE, SLURM_CPUS_PER_TASK) if SLURM_CPUS_PER_TASK > 1 else SLURM_CPUS_ON_NODE


def is_directory_empty(directory_path):
    try:
        return len(os.listdir(directory_path)) == 0
    except OSError:
        # Directory doesn't exist or can't be accessed
        return True


def filter_paths(collection_paths, index_paths):
    filtered_collection_paths = []
    filtered_index_paths = []
    for collection_path, index_path in zip(collection_paths, index_paths):
        compressed_index_path = index_path + ".tar.gz"
        uncompressed_index_path = index_path
        
        # Check if either compressed or uncompressed index exists
        compressed_exists = os.path.exists(compressed_index_path)
        uncompressed_exists = os.path.exists(uncompressed_index_path)
        
        if compressed_exists or uncompressed_exists:
            if compressed_exists:
                print(f"Skipping {collection_path} - compressed index already exists at {compressed_index_path}")
            if uncompressed_exists:
                print(f"Skipping {collection_path} - uncompressed index already exists at {uncompressed_index_path}")
            continue
        
        # Check if current user has write permissions to the collection_path
        try:
            if os.access(collection_path, os.W_OK):
                filtered_collection_paths.append(collection_path)
                filtered_index_paths.append(index_path)
            else:
                # print(f"Skipping {collection_path} - no write permissions")
                continue
        except (OSError, FileNotFoundError):
            print(f"Error checking permissions of {collection_path}")
            continue
    return filtered_collection_paths, filtered_index_paths


def create_build_index_commands(collection_paths, index_paths, threads_per_index=None):
    """
    Create build index commands for Pyserini.
    
    Args:
        collection_paths: List of collection paths to index
        index_paths: List of output index paths
        threads_per_index: Number of threads per index. If None, uses 1
    """
    if threads_per_index is None:
        threads_per_index = 1
    
    commands = []
    for collection_path, index_path in zip(collection_paths, index_paths):
        command = [
            "python", "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", collection_path,
            "--index", index_path,
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", str(threads_per_index),
            "--storePositions",
            "--storeDocvectors",
            "--storeRaw",
        ]
        commands.append(command)
    return commands


def build_index(command):
    try:
        result = subprocess.run(command, capture_output=True, check=True, text=True)
        # Get memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        print(f"Memory Usage: {memory_info.rss / 1024 / 1024 / 1024:.3f} GB")

        return(command, result.stdout)    
    except subprocess.CalledProcessError as e:
        return(command, f"Failed to build index with ERROR: {e.stderr}")


def build_indexes(collection_paths, index_paths, threads_per_index=1, max_parallel_jobs=None, overwrite=False):
    """
    Build indexes with configurable threading.
    
    Args:
        collection_paths: List of collection paths to index
        index_paths: List of output index paths
        threads_per_index: Number of threads per index
        max_parallel_jobs: Maximum number of parallel jobs. If None, calculated from available CPUs
        overwrite: Whether to overwrite existing indexes (default: False)
    """
    # Only make indexes that don't exist (unless overwrite is True)
    if not overwrite:
        collection_paths, index_paths = filter_paths(collection_paths, index_paths)
    else:
        print("Overwrite mode: skipping filter step, will process all collections")
    print(f"Filtered collection paths: {collection_paths}")
    commands = create_build_index_commands(collection_paths, index_paths, threads_per_index)
    # # Uncomment for testing and debugging
    # commands = commands[0:2]
    # command, output = build_index(commands[0])
    # print(f"Completed command: {' '.join(command)}")
    # print(f"Output: {output}")
    
    # Calculate max parallel jobs if not specified
    # max_parallel_jobs * threads_per_index <= TOTAL_CPUS
    if max_parallel_jobs is None:
        max_parallel_jobs = max(1, TOTAL_CPUS // threads_per_index)
    
    print(f"Created commands to build indexes")
    print(f"Threads per index: {threads_per_index}")
    print(f"Max parallel jobs: {max_parallel_jobs}")
    print(f"Total available CPUs: {TOTAL_CPUS}")
    print(f"Sample command: {' '.join(commands[0])}")
    print(f"Total commands: {len(commands)}")
    
    # Use the calculated max parallel jobs
    max_workers = min(max_parallel_jobs, len(commands))
    print(f"Max workers: {max_workers}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(build_index, command) for command in commands]
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            command, output = future.result()
            print(f"Completed command: {' '.join(command)}")
            print(f"Output: {output}")
            print()


def search(index_path, query, k=10, date_filter=None, field_filters=None, must_have_fields=None, include_null_dates=True, allow_null_for_sources=None, metadata_filter_policy: Optional[MetadataFilterPolicy] = None):
    """
    Search the BM25 index with optional filtering.
    Returns SearchResult objects for consistency with open search.
    
    Args:
        index_path: Path to the BM25 index
        query: Search query string
        k: Number of results to return (default: 10)
        date_filter: Dict with 'field' and 'before'/'after' keys for date filtering
                    e.g., {'field': 'published_epoch_seconds', 'before': 1609459200}
        field_filters: Dict of field-value pairs for exact matching
                      e.g., {'source': 'google_scholar', 'subdomain': 'bandit'}
        must_have_fields: List of fields that must be present (not null/empty)
                         e.g., ['published_epoch_seconds', 'cited_by']
        include_null_dates: If True, include documents with null dates when date_filter is applied.
                           If False, only return documents that satisfy the date filter.
                           Default: True (inclusive behavior)
        allow_null_for_sources: List of source values for which null dates are allowed
                               e.g., ['google_scholar'] allows null dates for google_scholar results
    
    Returns:
        List of SearchResult objects
    """
    searcher = LuceneSearcher(index_path)
    
    # Note: Field filters and must-have fields are applied post-search since --storeRaw
    # doesn't index individual fields for filtering
    
    # Execute search (without field filters since they're not indexed with --storeRaw)
    # Only apply text query and date filters in Lucene
    text_query = query
    if date_filter:
        field = date_filter['field']
        date_conditions = []
        
        if 'before' in date_filter:
            before_timestamp = date_filter['before']
            # Subtract 1 to exclude documents published on the resolution date
            # Lucene range queries are inclusive, so [* TO timestamp-1] excludes the resolution date
            exclusive_timestamp = before_timestamp - 1
            date_conditions.append(f"{field}:[* TO {exclusive_timestamp}]")
        
        if 'after' in date_filter:
            after_timestamp = date_filter['after']
            date_conditions.append(f"{field}:[{after_timestamp} TO *]")
        
        if date_conditions:
            # If allow_null_for_sources is specified, we need to include null dates
            # in the Lucene query and filter by source in post-processing
            if allow_null_for_sources or include_null_dates:
                # Include documents that either satisfy the date filter OR have null dates
                date_query = f"({' AND '.join(date_conditions)}) OR -{field}:*"
            else:
                # Strict mode: only documents that satisfy the date filter
                date_query = f"({' AND '.join(date_conditions)})"
            
            text_query += f" AND ({date_query})"
    
    search_type = "metadocuments" if "metadocuments" in (index_path or "") else "task_specific_documents"
    policy_for_search = (
        metadata_filter_policy
        if metadata_filter_policy and metadata_filter_policy.applies_to(search_type)
        else None
    )
    target_k = k
    k = compute_prefetch_k(k, policy_for_search)

    hits = searcher.search(text_query, k=max(k, target_k) * 3)
    
    # Convert hits to result dictionaries with full metadata and content
    # With --storeRaw, get the complete document from raw JSON
    results = []
    for i, hit in enumerate(hits):
        # Get the full document from raw JSON using --storeRaw
        raw_doc = searcher.doc(hit.docid)
        raw_content = raw_doc.raw()
        doc_data = json.loads(raw_content)
        
        # Apply post-search filtering
        if field_filters:
            skip_doc = False
            for field, value in field_filters.items():
                if doc_data.get(field) != value:
                    skip_doc = True
                    break
            if skip_doc:
                continue
        
        # Handle source-based null date filtering
        # If allow_null_for_sources is specified, exclude null-date docs from non-allowed sources
        if allow_null_for_sources and date_filter:
            date_field = date_filter['field']
            doc_date = doc_data.get(date_field)
            if doc_date is None or doc_date == '':
                # This doc has a null date - only keep if source is allowed
                doc_source = doc_data.get('source', '')
                if doc_source not in allow_null_for_sources:
                    continue  # Skip null-date docs from non-allowed sources
        
        if must_have_fields:
            skip_doc = False
            for field in must_have_fields:
                field_value = doc_data.get(field)
                if not field_value or (isinstance(field_value, str) and field_value.strip() == ''):
                    skip_doc = True
                    break
            if skip_doc:
                continue
        
        # Determine search type based on index path
        is_metadocument = "metadocuments" in index_path
        
        # Create SearchResult object based on document type
        if is_metadocument:
            # Metadocument format
            title = doc_data.get('title', '')
            url = doc_data.get('downloaded_link', doc_data.get('article_link', ''))
            snippet = doc_data.get('snippet', '')
            published_date = doc_data.get('published_date', '')
            
            # Create metadata with full document data
            metadata = {
                'id': hit.docid,
                'score': hit.score,
                **doc_data  # Include all document fields
            }
        else:
            # Docket file format
            title = doc_data.get('filename', '')
            url = doc_data.get('filename', '')  # Use filename as URL for docket files
            snippet = doc_data.get('snippet', '')
            published_date = doc_data.get('published_date', '')
            
            # Create metadata with full document data
            metadata = {
                'id': hit.docid,
                'score': hit.score,
                **doc_data  # Include all document fields
            }
        
        # Create SearchResult object
        search_result = SearchResult(
            title=title,
            url=url,
            snippet=snippet,
            source="closed_search",
            published_date=published_date,
            metadata=metadata,
            result_id=hit.docid
        )
        
        results.append(search_result)
        
        # Stop when we have enough results
        if len(results) >= k:
            break

    if policy_for_search:
        results, _ = apply_metadata_filters(query, results, search_type, policy_for_search)
    
    return results[:target_k]
