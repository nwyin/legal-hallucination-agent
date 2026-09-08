import os
import json
import logging
import numpy as np
import faiss
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Set offline mode for HuggingFace to prevent API calls
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle
from typing import List, Dict, Any, Optional
from readability import Readability
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, Range, SearchParams, MatchValue, MatchText, MatchAny, PointStruct
import requests
import socket

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
)

# No thread locks needed - using process-based parallelism


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_directory_empty(directory_path):
    try:
        return len(os.listdir(directory_path)) == 0
    except OSError:
        # Directory doesn't exist or can't be accessed
        return True


def check_internet_connection():
    try:
        # Try to connect to a reliable host
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False


def is_model_cached(model_name):
    try:
        # Check both custom cache folder and default cache folder
        cache_folders = []
        
        # Custom cache folder if set
        custom_cache = os.environ.get("SENTENCE_TRANSFORMERS_HOME")
        if custom_cache:
            cache_folders.append(custom_cache)
        
        # Default cache folder (SentenceTransformers default)
        default_cache = os.path.expanduser("~/.cache/torch/sentence_transformers")
        if os.path.exists(default_cache):
            cache_folders.append(default_cache)
        
        # Check all possible cache locations
        for cache_folder in cache_folders:
            # Try both SentenceTransformers format and HuggingFace format
            model_paths = [
                os.path.join(cache_folder, model_name.replace("/", "--")),  # SentenceTransformers format
                os.path.join(cache_folder, f"models--{model_name.replace('/', '--')}")  # HuggingFace format
            ]
            
            for model_path in model_paths:
                if os.path.exists(model_path) and os.path.isdir(model_path):
                    # Check if it has the necessary files (config.json is always present)
                    if os.path.exists(os.path.join(model_path, "config.json")):
                        return True
                    # For HuggingFace format, check snapshots directory
                    snapshots_path = os.path.join(model_path, "snapshots")
                    if os.path.exists(snapshots_path):
                        # Look for any snapshot directory with config.json
                        for snapshot in os.listdir(snapshots_path):
                            snapshot_path = os.path.join(snapshots_path, snapshot)
                            if os.path.isdir(snapshot_path) and os.path.exists(os.path.join(snapshot_path, "config.json")):
                                return True
        
        return False
    except Exception:
        return False


# =============================================================================
# MODEL HELPERS
# =============================================================================

NEMOTRON_EMBED_KEY = "llama-embed-nemotron"
QWEN3_EMBED_KEY = "Qwen3-Embedding"
DEWEY_EMBED_KEY = "dewey_en_beta"
MIXEDBREAD_EMBED_KEY = "mixedbread-ai"
E5_MISTRAL_EMBED_KEY = "e5-mistral-7b-instruct"


def _is_nemotron_model(model_name: str) -> bool:
    return NEMOTRON_EMBED_KEY in (model_name or "")


def _is_qwen3_embedding_model(model_name: str) -> bool:
    return QWEN3_EMBED_KEY in (model_name or "")


def _is_dewey_model(model_name: str) -> bool:
    return DEWEY_EMBED_KEY in (model_name or "")


def _is_mixedbread_model(model_name: str) -> bool:
    return MIXEDBREAD_EMBED_KEY in (model_name or "")


def _is_e5_mistral_model(model_name: str) -> bool:
    return E5_MISTRAL_EMBED_KEY in (model_name or "")


def _get_model_max_seq_length(model_name: str) -> int:
    """
    Get the practical max sequence length for a model.
    
    NOTE: These values are hard-set by us based on analysis of the 75th percentile 
    document lengths in our collections, NOT the theoretical max_position_embeddings 
    from the model config.json. This balances capturing most document content while 
    keeping GPU memory usage reasonable for batch processing.
    
    Theoretical max_position_embeddings from config.json:
    - dewey_en_beta: 131072 (128K)
    - llama-embed-nemotron-8b: 131072 (128K) 
    - Qwen3-Embedding-8B: 40960 (~40K)
    - e5-mistral-7b-instruct: 32768 (32K)
    - mxbai-embed-large-v1: 512
    """
    if _is_dewey_model(model_name):
        return 32768
    elif _is_nemotron_model(model_name):
        return 32768
    elif _is_qwen3_embedding_model(model_name):
        return 32768
    elif _is_e5_mistral_model(model_name):
        return 32768
    elif _is_mixedbread_model(model_name):
        return 512
    else:
        return 512  # Conservative default


def _pre_truncate_texts(texts: List[str], model_name: str, chars_per_token: float = 3.5) -> List[str]:
    """
    Pre-truncate texts before tokenization to avoid memory issues with very long documents.
    
    The tokenizer processes the entire input text before truncation, which can cause
    CUDA memory errors for extremely long documents (e.g., 137K words). This function
    truncates the text at the character level before tokenization.
    
    Args:
        texts: List of text strings to truncate
        model_name: Model name to determine max sequence length
        chars_per_token: Estimated characters per token (conservative estimate)
    
    Returns:
        List of truncated text strings
    """
    max_seq_length = _get_model_max_seq_length(model_name)
    # Use conservative estimate: ~3.5 chars per token for English text
    max_chars = int(max_seq_length * chars_per_token)
    
    truncated = []
    truncation_count = 0
    for text in texts:
        if len(text) > max_chars:
            truncated.append(text[:max_chars])
            truncation_count += 1
        else:
            truncated.append(text)
    
    if truncation_count > 0:
        print(f"Pre-truncated {truncation_count}/{len(texts)} documents to ~{max_chars} chars (max_seq_length={max_seq_length})")
    
    return truncated


def _call_with_optional_kwargs(fn, texts, **kwargs):
    try:
        return fn(texts, **kwargs)
    except TypeError:
        return fn(texts)


def _average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    attention_mask = attention_mask.unsqueeze(-1).type_as(last_hidden_states)
    last_hidden_states = last_hidden_states.to(torch.float32)
    masked_hidden = last_hidden_states * attention_mask
    summed = masked_hidden.sum(dim=1)
    counts = attention_mask.sum(dim=1).clamp(min=1e-6)
    averaged = summed / counts
    return F.normalize(averaged, dim=-1)


class NemotronEmbeddingModel:
    """Manual HuggingFace-based encoder for nvidia/llama-embed-nemotron-8b."""

    DEFAULT_INSTRUCTION = (
        "Given a legal research question, retrieve supporting passages."
    )

    def __init__(self, model_name: str, cache_folder: Optional[str], device: Optional[str]):
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = 4096
        self.max_seq_length = 32768
        self.query_instruction = os.environ.get(
            "NEMOTRON_QUERY_INSTRUCTION",
            self.DEFAULT_INSTRUCTION,
        )

        print(f"Initializing NemotronEmbeddingModel on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
            cache_dir=cache_folder,
            local_files_only=True,
        )
        attn_impl = "flash_attention_2" if self.device.startswith("cuda") and torch.cuda.is_available() else "eager"
        # Use float16 for memory efficiency (8B model needs ~16GB with float16 vs ~32GB with float32)
        # float16 works reliably on both CPU and CUDA without numpy conversion issues
        torch_dtype = torch.float16
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_folder,
            attn_implementation=attn_impl,
            torch_dtype=torch_dtype,
            local_files_only=True,
        ).to(self.device).eval()
        # Save embedding dimension for empty batches
        self.embedding_dim = getattr(self.model.config, "hidden_size", None) or getattr(self.model.config, "dim", 4096)

    def encode_documents(self, documents: List[str], batch_size: int = 8) -> np.ndarray:
        if not documents:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        texts = _pre_truncate_texts(documents, self.model_name)
        return self._encode_texts(texts, batch_size=batch_size)

    def encode_queries(self, queries: List[str], batch_size: int = 8) -> np.ndarray:
        if not queries:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        instructed = [
            f"Instruct: {self.query_instruction}\nQuery: {query}"
            for query in queries
        ]
        texts = _pre_truncate_texts(instructed, self.model_name)
        return self._encode_texts(texts, batch_size=batch_size)

    def _encode_texts(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        embeddings = []
        total = len(texts)
        for start in range(0, total, batch_size):
            batch_texts = texts[start:start + batch_size]
            tokenized = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            attention_mask = tokenized["attention_mask"]
            with torch.no_grad():
                outputs = self.model(**tokenized)
            pooled = _average_pool(outputs.last_hidden_state, attention_mask)
            # Convert to float32 before numpy (bfloat16/float16 not directly supported by numpy)
            embeddings.append(pooled.cpu().to(torch.float32).numpy().astype(np.float32))
        return np.vstack(embeddings)


def _last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class QwenEmbeddingModel:
    """Manual HuggingFace-based encoder for Qwen/Qwen3-Embedding-8B."""

    DEFAULT_INSTRUCTION = (
        "Given a legal research question, retrieve supporting passages."
    )

    def __init__(self, model_name: str, cache_folder: Optional[str], device: Optional[str]):
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = 8192  # Qwen3-Embedding max_length
        self.max_seq_length = 32768  # For pre-truncation
        self.query_instruction = os.environ.get(
            "QWEN_QUERY_INSTRUCTION",
            self.DEFAULT_INSTRUCTION,
        )

        print(f"Initializing QwenEmbeddingModel on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
            cache_dir=cache_folder,
            local_files_only=True,
        )
        # Try FlashAttention 2 if available and on CUDA
        attn_impl = "eager"
        use_flash_attention = False
        if self.device.startswith("cuda") and torch.cuda.is_available():
            try:
                import flash_attn
                attn_impl = "flash_attention_2"
                use_flash_attention = True
            except ImportError:
                pass
        
        # Use float16 for memory efficiency (8B model needs ~16GB with float16 vs ~32GB with float32)
        # float16 works reliably on both CPU and CUDA
        torch_dtype = torch.float16
        
        # Initialize model - for FlashAttention, initialize directly on GPU
        if use_flash_attention and self.device.startswith("cuda"):
            # FlashAttention requires GPU initialization - use .cuda() directly
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=cache_folder,
                attn_implementation=attn_impl,
                torch_dtype=torch_dtype,
                local_files_only=True,
            ).cuda().eval()
        else:
            # For eager attention, can initialize on CPU then move
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=cache_folder,
                attn_implementation=attn_impl,
                torch_dtype=torch_dtype,
                local_files_only=True,
            ).to(self.device).eval()
        
        # Save embedding dimension for empty batches
        self.embedding_dim = getattr(self.model.config, "hidden_size", None) or getattr(self.model.config, "dim", 2048)

    def set_query_instruction(self, instruction: str):
        self.query_instruction = instruction

    def _get_detailed_instruct(self, query: str) -> str:
        return f'Instruct: {self.query_instruction}\nQuery:{query}'

    def encode_documents(self, documents: List[str], batch_size: int = 8) -> np.ndarray:
        if not documents:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        texts = _pre_truncate_texts(documents, self.model_name)
        return self._encode_texts(texts, batch_size=batch_size)

    def encode_queries(self, queries: List[str], batch_size: int = 8) -> np.ndarray:
        if not queries:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        instructed = [self._get_detailed_instruct(query) for query in queries]
        texts = _pre_truncate_texts(instructed, self.model_name)
        return self._encode_texts(texts, batch_size=batch_size)

    def _encode_texts(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        embeddings = []
        total = len(texts)
        for start in tqdm(range(0, total, batch_size), desc="Encoding", unit="batch", leave=False):
            batch_texts = texts[start:start + batch_size]
            tokenized = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            attention_mask = tokenized["attention_mask"]
            with torch.no_grad():
                outputs = self.model(**tokenized)
            # Use last token pooling for Qwen
            pooled = _last_token_pool(outputs.last_hidden_state, attention_mask)
            # Normalize embeddings
            pooled = F.normalize(pooled, p=2, dim=1)
            # Convert to float32 before numpy (bfloat16/float16 not directly supported by numpy)
            embeddings.append(pooled.cpu().to(torch.float32).numpy().astype(np.float32))
        return np.vstack(embeddings)


def _cls_pool(outputs: torch.Tensor, attention_mask: torch.Tensor, strategy: str = 'cls') -> torch.Tensor:
    if strategy == 'cls':
        return outputs[:, 0]  # Take first token (CLS token)
    elif strategy == 'mean':
        # Mean pooling with attention mask
        masked = outputs * attention_mask[:, :, None]
        summed = masked.sum(dim=1)
        counts = attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return summed / counts
    else:
        raise NotImplementedError(f"Pooling strategy '{strategy}' not implemented")


class DeweyEmbeddingModel:
    """HuggingFace-based encoder for infgrad/dewey_en_beta using model.encode() method."""

    # Dewey retrieval prompts from README
    RETRIEVE_Q_PROMPT = "<|START_INSTRUCTION|>Answer the question<|END_INSTRUCTION|>"
    RETRIEVE_P_PROMPT = "<|START_INSTRUCTION|>Candidate document<|END_INSTRUCTION|>"

    def __init__(self, model_name: str, cache_folder: Optional[str], device: Optional[str]):
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_seq_length = 32768  # 32 * 1024 as per README
        
        print(f"Initializing DeweyEmbeddingModel on device: {self.device}")
        print("Using HuggingFace AutoModel with custom encode() method")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_folder,
            local_files_only=True
        )
        
        # Load model
        attn_impl = "flash_attention_2" if self.device.startswith("cuda") and torch.cuda.is_available() else "eager"
        torch_dtype = torch.bfloat16 if self.device.startswith("cuda") and torch.cuda.is_available() else torch.float32
        
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_folder,
            attn_implementation=attn_impl,
            torch_dtype=torch_dtype,
            local_files_only=True
        ).to(self.device)
        
        if torch_dtype == torch.bfloat16:
            self.model = self.model.bfloat16()
        
        self.model.eval()
        
        # Attach tokenizer to model (required by Dewey's encode method)
        self.model.tokenizer = self.tokenizer
        
        # Embedding dimension is 2048 for Dewey (ModernBERT-large)
        self.embedding_dim = 2048

    def _transform_query(self, query: str) -> str:
        return f"{self.RETRIEVE_Q_PROMPT}{query}"

    def _transform_document(self, document: str) -> str:
        return f"{self.RETRIEVE_P_PROMPT}{document}"

    def encode_documents(self, documents: List[str], batch_size: int = 8) -> np.ndarray:
        if not documents:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        
        # Pre-truncate to avoid memory issues
        texts = _pre_truncate_texts(documents, self.model_name)
        
        # Use Dewey's encode method with chunk_size=-1 for single vector
        # Returns: (List[np.ndarray], List[spans]) where each array is shape (2, 2048)
        use_cuda = self.device.startswith("cuda")
        encoded_results = self.model.encode(
            sentences=[self._transform_document(doc) for doc in texts],
            use_cuda=use_cuda,
            show_progress_bar=False,
            chunk_size=-1,  # Single vector mode
            chunk_overlap=32,
            convert_to_tensor=False,
            max_seq_length=self.max_seq_length,
            batch_size=batch_size,
            normalize_embeddings=True,
            prompt="",  # Already added to text
            fast_chunk=False
        )
        
        # Extract the vectors list (first element of tuple)
        if isinstance(encoded_results, tuple):
            encoded = encoded_results[0]  # List of arrays
        else:
            encoded = encoded_results
        
        # Extract mean vectors (index 1) from each result
        # Each result is shape (2, 2048): [CLS vector, mean vector]
        embeddings = []
        for result in encoded:
            if isinstance(result, np.ndarray) and result.ndim == 2:
                # Take the mean vector (index 1) for long text retrieval
                embeddings.append(result[1, :])  # Shape (2048,)
            else:
                # Fallback if unexpected format
                embeddings.append(result.flatten() if hasattr(result, 'flatten') else result)
        
        # Stack and ensure correct shape
        embeddings = np.vstack(embeddings)
        return embeddings.astype(np.float32)

    def encode_queries(self, queries: List[str], batch_size: int = 8) -> np.ndarray:
        if not queries:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        
        # Pre-truncate to avoid memory issues
        texts = _pre_truncate_texts(queries, self.model_name)
        
        # Use Dewey's encode method with chunk_size=-1 for single vector
        use_cuda = self.device.startswith("cuda")
        encoded_results = self.model.encode(
            sentences=[self._transform_query(query) for query in texts],
            use_cuda=use_cuda,
            show_progress_bar=False,
            chunk_size=-1,  # Single vector mode
            chunk_overlap=32,
            convert_to_tensor=False,
            max_seq_length=self.max_seq_length,
            batch_size=batch_size,
            normalize_embeddings=True,
            prompt="",  # Already added to text
            fast_chunk=False
        )
        
        # Extract the vectors list (first element of tuple)
        if isinstance(encoded_results, tuple):
            encoded = encoded_results[0]  # List of arrays
        else:
            encoded = encoded_results
        
        # Extract mean vectors (index 1) from each result
        embeddings = []
        for result in encoded:
            if isinstance(result, np.ndarray) and result.ndim == 2:
                # Take the mean vector (index 1) for queries
                embeddings.append(result[1, :])  # Shape (2048,)
            else:
                embeddings.append(result.flatten() if hasattr(result, 'flatten') else result)
        
        # Stack and ensure correct shape
        embeddings = np.vstack(embeddings)
        return embeddings.astype(np.float32)


class E5MistralEmbeddingModel:
    """Manual HuggingFace-based encoder for intfloat/e5-mistral-7b-instruct."""

    DEFAULT_TASK = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )

    def __init__(self, model_name: str, cache_folder: Optional[str], device: Optional[str]):
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = 4096  # e5-mistral-7b-instruct max_length
        self.max_seq_length = 32768  # For pre-truncation
        self.task_description = os.environ.get(
            "E5_MISTRAL_TASK_DESCRIPTION",
            self.DEFAULT_TASK,
        )

        print(f"Initializing E5MistralEmbeddingModel on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_folder,
            local_files_only=True,
        )
        attn_impl = "flash_attention_2" if self.device.startswith("cuda") and torch.cuda.is_available() else "eager"
        torch_dtype = torch.float16 if self.device.startswith("cuda") and torch.cuda.is_available() else torch.float32
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_folder,
            attn_implementation=attn_impl,
            torch_dtype=torch_dtype,
            local_files_only=True,
        ).to(self.device).eval()
        # Save embedding dimension for empty batches
        self.embedding_dim = getattr(self.model.config, "hidden_size", None) or getattr(self.model.config, "dim", 4096)

    def _get_detailed_instruct(self, query: str) -> str:
        return f'Instruct: {self.task_description}\nQuery: {query}'

    def encode_documents(self, documents: List[str], batch_size: int = 8) -> np.ndarray:
        if not documents:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        texts = _pre_truncate_texts(documents, self.model_name)
        return self._encode_texts(texts, batch_size=batch_size)

    def encode_queries(self, queries: List[str], batch_size: int = 8) -> np.ndarray:
        if not queries:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        instructed = [self._get_detailed_instruct(query) for query in queries]
        texts = _pre_truncate_texts(instructed, self.model_name)
        return self._encode_texts(texts, batch_size=batch_size)

    def _encode_texts(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        embeddings = []
        total = len(texts)
        for start in range(0, total, batch_size):
            batch_texts = texts[start:start + batch_size]
            tokenized = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            attention_mask = tokenized["attention_mask"]
            with torch.no_grad():
                outputs = self.model(**tokenized)
            # Use last token pooling for e5-mistral (same as Qwen)
            pooled = _last_token_pool(outputs.last_hidden_state, attention_mask)
            # Normalize embeddings
            pooled = F.normalize(pooled, p=2, dim=1)
            # Convert to float32 before numpy (bfloat16/float16 not directly supported by numpy)
            embeddings.append(pooled.cpu().to(torch.float32).numpy().astype(np.float32))
        return np.vstack(embeddings)


class MixedbreadEmbeddingModel:
    """Manual HuggingFace-based encoder for mixedbread-ai/mxbai-embed-large-v1."""

    def __init__(self, model_name: str, cache_folder: Optional[str], device: Optional[str]):
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = 512  # Conservative default, will use model's actual max
        self.max_seq_length = 512  # For pre-truncation (mixedbread has shorter context)
        self.pooling_strategy = os.environ.get("MIXEDBREAD_POOLING_STRATEGY", "cls")  # 'cls' or 'mean'

        print(f"Initializing MixedbreadEmbeddingModel on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_folder,
            local_files_only=True,
        )
        attn_impl = "flash_attention_2" if self.device.startswith("cuda") and torch.cuda.is_available() else "eager"
        torch_dtype = torch.float16 if self.device.startswith("cuda") and torch.cuda.is_available() else torch.float32
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_folder,
            attn_implementation=attn_impl,
            torch_dtype=torch_dtype,
            local_files_only=True,
        ).to(self.device).eval()
        # Save embedding dimension for empty batches
        self.embedding_dim = getattr(self.model.config, "hidden_size", None) or getattr(self.model.config, "dim", 1024)

    def _transform_query(self, query: str) -> str:
        return f'Represent this sentence for searching relevant passages: {query}'

    def encode_documents(self, documents: List[str], batch_size: int = 8) -> np.ndarray:
        if not documents:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        texts = _pre_truncate_texts(documents, self.model_name)
        return self._encode_texts(texts, batch_size=batch_size)

    def encode_queries(self, queries: List[str], batch_size: int = 8) -> np.ndarray:
        if not queries:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        transformed = [self._transform_query(query) for query in queries]
        texts = _pre_truncate_texts(transformed, self.model_name)
        return self._encode_texts(texts, batch_size=batch_size)

    def _encode_texts(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        embeddings = []
        total = len(texts)
        for start in range(0, total, batch_size):
            batch_texts = texts[start:start + batch_size]
            tokenized = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            attention_mask = tokenized["attention_mask"]
            with torch.no_grad():
                outputs = self.model(**tokenized)
            # Use CLS or mean pooling for mixedbread (same as Dewey)
            pooled = _cls_pool(outputs.last_hidden_state, attention_mask, strategy=self.pooling_strategy)
            # Normalize embeddings
            pooled = F.normalize(pooled, p=2, dim=1)
            # Convert to float32 before numpy (bfloat16/float16 not directly supported by numpy)
            embeddings.append(pooled.cpu().to(torch.float32).numpy().astype(np.float32))
        return np.vstack(embeddings)


def _encode_documents_for_model(model, model_name, documents, batch_size=32):
    # Pre-truncate documents to avoid tokenizer memory issues with very long texts
    documents = _pre_truncate_texts(documents, model_name)

    # Custom encoder path (Nemotron, Qwen, Dewey use HuggingFace Transformers directly)
    if hasattr(model, "encode_documents"):
        embeddings = model.encode_documents(documents, batch_size=batch_size)
        return np.asarray(embeddings)
    
    if _is_nemotron_model(model_name):
        encode_fn = getattr(model, "encode_document", None)
        if encode_fn:
            embeddings = _call_with_optional_kwargs(
                encode_fn,
                documents,
                batch_size=batch_size,
                show_progress_bar=True,
            )
        else:
            embeddings = model.encode(
                documents,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
    else:
        encode_kwargs = {
            "batch_size": batch_size,
            "show_progress_bar": True,
            "convert_to_numpy": True,
            "normalize_embeddings": True
        }
        if _is_qwen3_embedding_model(model_name):
            embeddings = model.encode(documents, **encode_kwargs)
        else:
            embeddings = model.encode(documents, **encode_kwargs)
    return np.asarray(embeddings)


def _encode_queries_for_model(model, model_name, queries, batch_size=32):
    # All models now use custom HuggingFace Transformers encoders with encode_queries method
    if hasattr(model, "encode_queries"):
        embeddings = model.encode_queries(queries, batch_size=batch_size)
        return np.asarray(embeddings)

    # Fallback for unknown models using SentenceTransformer
    embeddings = model.encode(queries)
    return np.asarray(embeddings)


# =============================================================================
# DOCUMENT LOADING AND EMBEDDING GENERATION
# =============================================================================

def load_documents(collection_path):
    documents = []
    
    if os.path.isfile(collection_path):
        # Single file case (backward compatibility with JSONL)
        with open(collection_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    doc = json.loads(line.strip())
                    documents.append(doc)
    elif os.path.isdir(collection_path):
        # Directory case - load all JSON files
        json_files = [f for f in os.listdir(collection_path) if f.endswith('.json')]
        print(f"Found {len(json_files)} JSON files in {collection_path}")
        for json_file in json_files:
            file_path = os.path.join(collection_path, json_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                    documents.append(doc)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading {json_file}: {e}")
                continue
    else:
        raise ValueError(f"Collection path must be a file or directory: {collection_path}")
    
    return documents


def generate_embeddings(documents, model_name, batch_size=32, dimensions=None, device=None):
    """
    Generate embeddings for documents using SentenceTransformers.

    Args:
        documents: List of documents (dicts with 'contents' field)
        model_name: HuggingFace model name for embeddings (or local path)
        batch_size: Batch size for embedding generation
        dimensions: Number of dimensions for embeddings (supports MRL for mixedbread-ai models)
        device: Device to use for computation ('cuda', 'cpu', or None for auto-detection)
    
    Returns:
        numpy array of embeddings
    """
    # Check if model is cached or if we have internet connection
    model_cached = is_model_cached(model_name)
    has_internet = check_internet_connection()
    
    if model_cached:
        print(f"Model '{model_name}' found in local cache")
    elif has_internet:
        print(f"Model '{model_name}' not cached, will download from HuggingFace")
    else:
        raise ConnectionError(
            f"Model '{model_name}' is not cached locally and no internet connection available. "
            "Please run setup_offline_search.py first or ensure internet connectivity."
        )
    
    # Auto-detect device if not specified
    if device is None:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Load model - use the unified loading function for consistency
    model = load_embedding_model(model_name, cache_folder=os.environ.get("SENTENCE_TRANSFORMERS_HOME"), device=device)
    
    # Handle MRL (Matryoshka Representation Learning) for mixedbread-ai models
    if "mixedbread-ai" in model_name and dimensions:
        print(f"Note: MRL dimension setting ({dimensions}) - model was loaded with full dimensions")
        # MRL is applied via truncate_dim during model instantiation, but we've already loaded
        # For future: could reload with truncate_dim if needed
    
    # Set max sequence length for specific models (practical values for GPU memory)
    if hasattr(model, "max_seq_length"):
        pass  # Custom models (e.g., NemotronEmbeddingModel) already set this
    elif _is_dewey_model(model_name):
        print(f"Setting max_seq_length to 32768 for dewey_en_beta")
        model.max_seq_length = 32768
    elif _is_e5_mistral_model(model_name):
        print(f"Setting max_seq_length to 32768 for e5-mistral-7b-instruct")
        model.max_seq_length = 32768
    elif _is_qwen3_embedding_model(model_name):
        print(f"Setting max_seq_length to 32768 for Qwen3-Embedding")
        model.max_seq_length = 32768
    
    # Extract contents from documents
    contents = [doc.get('contents', '') for doc in documents]
    
    if hasattr(model, "encode_documents"):
        print(f"Generating embeddings via custom encoder for {len(contents)} documents...")
        embeddings = model.encode_documents(contents, batch_size=batch_size)
    elif _is_dewey_model(model_name):
        print(f"Using Dewey retrieve_passage prompt for {len(contents)} documents...")
        embeddings = model.encode(
            contents, 
            batch_size=batch_size, 
            show_progress_bar=True,
            prompt_name="retrieve_passage"
        )
    else:
        print(f"Generating embeddings for {len(contents)} documents...")
        embeddings = _encode_documents_for_model(model, model_name, contents, batch_size=batch_size)
    
    return embeddings


# =============================================================================
# INDEX BUILDING FUNCTIONS
# =============================================================================

def build_faiss_index(embeddings, index_type="Flat", nlist=100):
    """
    Build Faiss index for dense retrieval.
    
    Args:
        embeddings: numpy array of embeddings
        index_type: Type of Faiss index ("IVFFlat", "Flat", "HNSW")
                   - "Flat": Exact search, best for small collections (<10k docs)
                   - "IVFFlat": Approximate search, good for large collections
                   - "HNSW": Approximate search, good for very large collections
        nlist: Number of clusters for IVFFlat (ignored for other types)
    
    Returns:
        Faiss index object
    """
    dimension = embeddings.shape[1]
    
    if index_type == "IVFFlat":
        # Create quantizer
        quantizer = faiss.IndexFlatL2(dimension)
        # Create IVF index
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        # Ensure embeddings are contiguous and float32
        embeddings_float32 = np.ascontiguousarray(embeddings.astype('float32'))
        # Train the index
        print("Training Faiss index...")
        index.train(embeddings_float32)
        # Add vectors to index
        print("Adding vectors to index...")
        index.add(embeddings_float32)
    elif index_type == "Flat":
        # Simple flat index
        index = faiss.IndexFlatIP(dimension)
        # Ensure embeddings are contiguous and float32
        embeddings_float32 = np.ascontiguousarray(embeddings.astype('float32'))
        # Normalize embeddings for cosine similarity with inner product
        faiss.normalize_L2(embeddings_float32)
        index.add(embeddings_float32)
    elif index_type == "HNSW":
        # Hierarchical Navigable Small World index
        index = faiss.IndexHNSWFlat(dimension, 32)  # 32 is the M parameter
        # Ensure embeddings are contiguous and float32
        embeddings_float32 = np.ascontiguousarray(embeddings.astype('float32'))
        index.add(embeddings_float32)
    else:
        raise ValueError(f"Unsupported index type: {index_type}")
    
    return index


def build_qdrant_index(embeddings, documents, index_path, collection_name, overwrite=False):
    """
    Build Qdrant index for dense retrieval with metadata filtering support.
    
    Args:
        embeddings: numpy array of embeddings
        documents: List of documents (dicts with metadata)
        index_path: Path to save the Qdrant index
        collection_name: Name for the Qdrant collection
        overwrite: Whether to overwrite existing collection (default: False)
    
    Returns:
        tuple: (client, documents) or (None, documents) if failed
    """
    # Initialize Qdrant client - store directly in index_path like Faiss
    client = QdrantClient(path=index_path)
    
    # Check if collection already exists
    try:
        collections = client.get_collections()
        collection_exists = any(c.name == collection_name for c in collections.collections)
        
        if collection_exists and not overwrite:
            print(f"Skipping - collection {collection_name} already exists")
            return None, documents
        elif collection_exists and overwrite:
            print(f"Overwrite mode: will rebuild collection {collection_name}")
            client.delete_collection(collection_name)
    except Exception as e:
        print(f"Error checking collection {collection_name}: {e}")
        return None, documents
    
    # Get embedding dimension
    dimension = len(embeddings[0])
    
    # Create collection
    try:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )
        print(f"Created collection: {collection_name}")
    except Exception as e:
        print(f"Error creating collection {collection_name}: {e}")
        return None, documents
    
    # Note: Payload indexes only work with Qdrant server, not local file-based client.
    # Filtering still works without indexes, just slower (linear scan).
    # For small collections (hundreds of docs), this is fine.
    
    # Normalize all embeddings for cosine similarity at once
    embeddings = embeddings.astype("float32")
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    
    # Prepare all points for Qdrant at once
    points = []
    for idx, (doc, embedding) in enumerate(zip(documents, embeddings)):
        points.append(PointStruct(
            id=idx,  # Use sequential integer ID (Qdrant requirement)
            vector=embedding.tolist(),
            payload=doc  # Use entire document as payload (includes original doc["id"])
        ))
    
    # Upload points to Qdrant using recommended method
    try:
        client.upload_points(
            collection_name=collection_name,
            points=points
        )
        print(f"Successfully uploaded {len(points)} documents to collection {collection_name}")
    except Exception as e:
        print(f"Error uploading to collection {collection_name}: {e}")
        return None, documents
    
    print(f"Successfully built Qdrant index for {len(documents)} documents in {collection_name}")
    return client, documents


def save_faiss_index(index, metadata, index_path):
    """
    Save Faiss index and metadata to disk.
    
    Args:
        index: Faiss index object
        metadata: List of document metadata
        index_path: Path to save the index
    """
    # Create directory if it doesn't exist
    os.makedirs(index_path, exist_ok=True)
    
    # Save Faiss index
    faiss_path = os.path.join(index_path, "faiss_index.bin")
    faiss.write_index(index, faiss_path)
    
    # Save metadata
    metadata_path = os.path.join(index_path, "metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Faiss index saved to {index_path}")


def load_dense_index(index_path):
    """
    Load Faiss index and metadata from disk.
    
    Args:
        index_path: Path to the saved index
    
    Returns:
        tuple: (index, metadata)
    """
    faiss_path = os.path.join(index_path, "faiss_index.bin")
    metadata_path = os.path.join(index_path, "metadata.pkl")
    
    if not os.path.exists(faiss_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Index files not found in {index_path}")
    
    # Load Faiss index
    index = faiss.read_index(faiss_path)
    
    # Load metadata
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return index, metadata


# =============================================================================
# SEARCH FUNCTIONS
# =============================================================================

def create_qdrant_filter(date_filter=None, field_filters=None, text_filters=None, must_have_fields=None, include_null_dates=True, allow_null_for_sources=None):
    """
    Create a Qdrant Filter object from the filter parameters.
    
    Args:
        date_filter: Dict with 'field' and 'before'/'after' keys for date filtering
        field_filters: Dict of field-value pairs for exact matching (uses MatchValue)
        text_filters: Dict of field-value pairs for text/substring matching (uses MatchText)
                     e.g., {"subdomain": "federalism"} matches "Federalism and state authority..."
        must_have_fields: List of fields that must be present (not null/empty)
        include_null_dates: If True, include documents with null dates when date_filter is applied
        allow_null_for_sources: List of source values (e.g., ["google_scholar"]) for which null dates 
                               are allowed. If provided, overrides include_null_dates with source-specific logic.
    
    Returns:
        Qdrant Filter object or None if no filters
    """
    conditions = []
    
    # Date filtering
    if date_filter:
        field = date_filter['field']
        date_conditions = []
        
        if 'before' in date_filter:
            before_timestamp = date_filter['before']
            # Use 'lt' (strictly less than) to exclude documents published on the resolution date
            # This prevents data leakage from documents published on the same day as the case decision
            date_conditions.append(FieldCondition(key=field, range=Range(lt=before_timestamp)))
        
        if 'after' in date_filter:
            after_timestamp = date_filter['after']
            date_conditions.append(FieldCondition(key=field, range=Range(gte=after_timestamp)))
        
        if date_conditions:
            # Handle null dates based on source if specified
            if allow_null_for_sources:
                # Allow null dates only for specific sources (e.g., google_scholar)
                # This creates: (date < before) OR (date IS NULL AND source IN allow_null_for_sources)
                for source in allow_null_for_sources:
                    null_with_source = Filter(must=[
                        FieldCondition(key=field, is_null=True),
                        FieldCondition(key="source", match=MatchValue(value=source))
                    ])
                    date_conditions.append(null_with_source)
            elif include_null_dates:
                # Include documents that either satisfy the date filter OR have null dates
                # For null dates, we need to check if the field is null
                null_condition = FieldCondition(key=field, is_null=True)
                date_conditions.append(null_condition)
            
            if len(date_conditions) == 1:
                conditions.extend(date_conditions)
            else:
                # Combine multiple date conditions with OR (satisfy date filter OR null)
                conditions.append(Filter(should=date_conditions))
    
    # Field filtering (exact match)
    if field_filters:
        for field, value in field_filters.items():
            # Skip None values - they can't be matched
            if value is None:
                continue
            # Ensure value is a valid type for Qdrant MatchValue (str, int, float, bool, or list)
            # Convert to string if it's not a supported type
            if not isinstance(value, (str, int, float, bool, list)):
                value = str(value)
            try:
                conditions.append(FieldCondition(key=field, match=MatchValue(value=value)))
            except Exception as e:
                logger.warning(f"Failed to create field filter for {field}={value}: {e}. Skipping this filter.")
                continue
    
    # Text filtering (substring/token match - useful for subdomain, title, etc.)
    if text_filters:
        for field, text in text_filters.items():
            conditions.append(FieldCondition(key=field, match=MatchText(text=text)))
    
    # Must have fields (existence check)
    if must_have_fields:
        for field in must_have_fields:
            # Check that field exists and is not null
            conditions.append(FieldCondition(key=field, is_null=False))
    
    if not conditions:
        return None
    
    if len(conditions) == 1:
        return Filter(must=conditions)
    else:
        return Filter(must=conditions)


def generate_embeddings_with_model(documents, model, model_name, batch_size=32, dimensions=None):
    """
    Generate embeddings for documents using a pre-loaded SentenceTransformer model.

    Args:
        documents: List of documents (dicts with 'contents' field)
        model: Pre-loaded SentenceTransformer model
        model_name: HuggingFace model name (for logging)
        batch_size: Batch size for embedding generation
        dimensions: Number of dimensions for embeddings (supports MRL for mixedbread-ai models)

    Returns:
        numpy array of embeddings
    """
    # Extract contents from documents
    contents = [doc.get('contents', '') for doc in documents]
    
    # Pre-truncate to avoid tokenizer memory issues with very long documents
    contents = _pre_truncate_texts(contents, model_name)
    
    if hasattr(model, "encode_documents"):
        embeddings = model.encode_documents(contents, batch_size=batch_size)
    else:
        embeddings = _encode_documents_for_model(model, model_name, contents, batch_size=batch_size)
    
    return embeddings


def load_embedding_model(model_name, cache_folder=os.environ.get("SENTENCE_TRANSFORMERS_HOME"), device=None):
    """
    Load and return an embedding model for reuse.
    
    Args:
        model_name: HuggingFace model name for query embedding
        cache_folder: Path to cache folder for SentenceTransformers
        device: Device to use for computation ('cuda', 'cpu', or None for auto-detection)
    
    Returns:
        Loaded SentenceTransformer model
    """
    # Check if model is cached or if we have internet connection
    model_cached = is_model_cached(model_name)
    has_internet = check_internet_connection()
    
    if model_cached:
        print(f"Model '{model_name}' found in local cache")
    elif has_internet:
        print(f"Model '{model_name}' not cached, will download from HuggingFace")
    else:
        raise ConnectionError(
            f"Model '{model_name}' is not cached locally and no internet connection available. "
            "Please run setup_offline_search.py first or ensure internet connectivity."
        )
    
    # Auto-detect device if not specified
    if device is None:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading embedding model '{model_name}' on device: {device}")

    # Use custom HuggingFace-based encoders for models ported to Transformers
    if _is_nemotron_model(model_name):
        return NemotronEmbeddingModel(model_name, cache_folder, device)
    
    if _is_qwen3_embedding_model(model_name):
        return QwenEmbeddingModel(model_name, cache_folder, device)
    
    if _is_dewey_model(model_name):
        return DeweyEmbeddingModel(model_name, cache_folder, device)
    
    if _is_e5_mistral_model(model_name):
        return E5MistralEmbeddingModel(model_name, cache_folder, device)
    
    if _is_mixedbread_model(model_name):
        return MixedbreadEmbeddingModel(model_name, cache_folder, device)

    # All models should now be using custom HuggingFace Transformers encoders
    # If we reach here, it's an unknown model - fall back to SentenceTransformer
    print(f"Warning: Unknown model '{model_name}', falling back to SentenceTransformer")
    print(f"About to create SentenceTransformer instance...")
    model = SentenceTransformer(model_name, cache_folder=cache_folder, device=device)
    print(f"SentenceTransformer model loaded successfully!")
    return model


def search_with_model(model, index_path, query_text, model_name, k=10, use_readability_reranking=True, initial_k=15, readability_bias: float = 0.4, backend="faiss", collection_name=None, date_filter=None, field_filters=None, text_filters=None, must_have_fields=None, include_null_dates=True, allow_null_for_sources=None, metadata_filter_policy: Optional[MetadataFilterPolicy] = None):
    """
    Search a dense index with a pre-loaded model.
    
    Args:
        model: Pre-loaded SentenceTransformer model
        index_path: Path to the saved index
        query_text: Text query
        model_name: HuggingFace model name (used for mixedbread-ai detection)
        k: Number of results to return
        use_readability_reranking: Whether to apply readability-based reranking (default: True)
        initial_k: Number of initial results to retrieve before reranking (default: 15)
        backend: Backend type ("faiss" or "qdrant")
        collection_name: Collection name for Qdrant (only used for backend="qdrant")
        date_filter: Dict with 'field' and 'before'/'after' keys for date filtering
                    e.g., {'field': 'published_epoch_seconds', 'before': 1609459200}
        field_filters: Dict of field-value pairs for exact matching (uses MatchValue)
                      e.g., {'source': 'google_scholar', 'link_name': 'Main Document'}
        text_filters: Dict of field-value pairs for text/substring matching (uses MatchText)
                     e.g., {'subdomain': 'federalism'} matches "Federalism and state authority..."
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
    # Determine search type for downstream metadata policies
    search_type = "metadocuments" if "metadocuments" in (index_path or "") else "task_specific_documents"
    target_k = k
    
    # No prefetch needed - filters are applied directly in Qdrant query if provided
    
    # Generate query embedding using pre-loaded model
    query_embedding = _encode_queries_for_model(model, model_name, [query_text], batch_size=8)
    
    # Search with larger initial set if reranking is enabled
    if use_readability_reranking:
        # Retrieve more results initially to have a larger pool for reranking
        search_k = max(initial_k, k * 3)  # At least 3x the final k, or initial_k
    else:
        search_k = k
    
    if backend == "faiss":
        # Load Faiss index and metadata
        index, metadata = load_dense_index(index_path)
        
        # Search Faiss index
        results = search_faiss_index(index, metadata, query_embedding, search_k, use_readability_reranking, readability_bias=readability_bias, date_filter=date_filter, field_filters=field_filters, must_have_fields=must_have_fields, include_null_dates=include_null_dates, allow_null_for_sources=allow_null_for_sources, query_text=query_text, index_path=index_path)
    
    elif backend == "qdrant":
        # Use provided collection name or derive from path
        if not collection_name:
            # Fallback: derive from index path
            collection_name = os.path.basename(index_path)
        
        # Search Qdrant index
        results = search_qdrant_index(index_path, collection_name, query_text, model, model_name, search_k, use_readability_reranking, readability_bias, date_filter=date_filter, field_filters=field_filters, text_filters=text_filters, must_have_fields=must_have_fields, include_null_dates=include_null_dates, allow_null_for_sources=allow_null_for_sources)
    
    else:
        raise ValueError(f"Unsupported backend: {backend}. Must be 'faiss' or 'qdrant'")

    # No post-processing needed - filters are applied directly in Qdrant query
    final_results = list(results)[:target_k]
    return final_results


def search(index_path, query_text, model_name, k=10, cache_folder=os.environ.get("SENTENCE_TRANSFORMERS_HOME"), use_readability_reranking=True, initial_k=15, readability_bias: float = 0.4, backend="faiss", collection_name=None, device=None, date_filter=None, field_filters=None, must_have_fields=None, include_null_dates=True, allow_null_for_sources=None, metadata_filter_policy: Optional[MetadataFilterPolicy] = None):
    """
    Search a dense index with a text query.
    
    Args:
        index_path: Path to the saved index
        query_text: Text query
        model_name: HuggingFace model name for query embedding
        k: Number of results to return
        cache_folder: Path to cache folder for SentenceTransformers
        use_readability_reranking: Whether to apply readability-based reranking (default: True)
        initial_k: Number of initial results to retrieve before reranking (default: 15)
        backend: Backend type ("faiss" or "qdrant")
        collection_name: Collection name for Qdrant (only used for backend="qdrant")
        device: Device to use for computation ('cuda', 'cpu', or None for auto-detection)
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
    # Load model using the new function
    model = load_embedding_model(model_name, cache_folder, device)
    
    # Use the new search_with_model function
    return search_with_model(
        model=model,
        index_path=index_path,
        query_text=query_text,
        model_name=model_name,
        k=k,
        use_readability_reranking=use_readability_reranking,
        initial_k=initial_k,
        readability_bias=readability_bias,
        backend=backend,
        collection_name=collection_name,
        date_filter=date_filter,
        field_filters=field_filters,
        must_have_fields=must_have_fields,
        include_null_dates=include_null_dates,
        allow_null_for_sources=allow_null_for_sources,
        metadata_filter_policy=metadata_filter_policy,
    )


def search_faiss_index(index, metadata, query_embedding, k=10, use_readability_reranking=True, readability_bias: float = 0.4, date_filter=None, field_filters=None, must_have_fields=None, include_null_dates=True, allow_null_for_sources=None, query_text="", index_path=""):
    """
    Search the dense index for similar documents.
    
    Args:
        index: Faiss index object
        metadata: List of document metadata
        query_embedding: Query embedding vector
        k: Number of results to return
        use_readability_reranking: Whether to apply readability-based reranking
        query_text: Original query text (for metadata)
        index_path: Path to index (for determining metadocument vs docket file format)
        allow_null_for_sources: List of source values for which null dates are allowed
    
    Returns:
        List of SearchResult objects
    """
    # Ensure query is 2D
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    # Convert to float32 and normalize if using IndexFlatIP
    query_float32 = query_embedding.astype('float32')
    if hasattr(index, 'metric_type') and index.metric_type == faiss.METRIC_INNER_PRODUCT:
        # Normalize query for cosine similarity with inner product
        faiss.normalize_L2(query_float32)
    
    # Search - get more results initially for post-filtering
    search_k = min(k * 3, len(metadata))  # Don't exceed available documents
    scores, indices = index.search(query_float32, search_k)
    
    # Debug: log initial search results
    initial_count = len([idx for idx in indices[0] if idx < len(metadata)])
    logger.debug(f"Initial search returned {initial_count} results (requested {search_k}, metadata has {len(metadata)} docs)")
    
    results = []
    date_filtered_count = 0
    field_filtered_count = 0
    null_date_excluded_count = 0
    
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(metadata):  # Valid index
            doc = metadata[idx]
            
            # Apply post-search filtering (since Faiss doesn't support native filtering)
            if date_filter:
                field = date_filter['field']
                doc_value = doc.get(field)
                
                if doc_value is None or doc_value == '':
                    # Handle null dates based on source if specified
                    if allow_null_for_sources:
                        # Allow null dates only for specific sources (e.g., google_scholar)
                        doc_source = doc.get('source', '')
                        if doc_source not in allow_null_for_sources:
                            null_date_excluded_count += 1
                            continue  # Skip null dates from non-allowed sources
                    elif not include_null_dates:
                        null_date_excluded_count += 1
                        continue  # Skip documents with null dates in strict mode
                else:
                    # Apply date filter - exclude documents published on or after the resolution date
                    if 'before' in date_filter and doc_value >= date_filter['before']:
                        date_filtered_count += 1
                        continue
                    if 'after' in date_filter and doc_value < date_filter['after']:
                        date_filtered_count += 1
                        continue
            
            if field_filters:
                skip_doc = False
                for field, value in field_filters.items():
                    if doc.get(field) != value:
                        skip_doc = True
                        break
                if skip_doc:
                    field_filtered_count += 1
                    continue
            
            if must_have_fields:
                skip_doc = False
                for field in must_have_fields:
                    field_value = doc.get(field)
                    if not field_value or (isinstance(field_value, str) and field_value.strip() == ''):
                        skip_doc = True
                        break
                if skip_doc:
                    continue
            
            # Extract document text and title
            doc_text = doc.get('contents', '')
            title = doc.get('title', 'N/A')
            
            if use_readability_reranking:
                # Apply readability reranking
                combined_score = apply_readability_reranking(float(score), doc_text, readability_bias)
            else:
                # Use only semantic similarity score
                combined_score = float(score)
            
            # Determine search type based on index path (same logic as BM25)
            is_metadocument = "metadocuments" in index_path
            
            # Create SearchResult object based on document type (same logic as BM25)
            if is_metadocument:
                # Metadocument format
                title = doc.get('title', '')
                url = doc.get('downloaded_link', doc.get('article_link', ''))
                snippet = doc.get('snippet', '')
                published_date = doc.get('published_date', '')
                
                # Create metadata with full document data
                metadata = {
                    'score': combined_score,
                    'original_score': float(score),
                    'doc_id': idx,
                    'retrieval_query': query_text,
                    'original_query': doc.get('query'),
                    **doc  # Include all document fields
                }
            else:
                # Docket file format
                title = doc.get('filename', '')
                url = doc.get('filename', '')  # Use filename as URL for docket files
                snippet = doc.get('snippet', '')
                published_date = doc.get('published_date', '')
                
                # Create metadata with full document data
                metadata = {
                    'score': combined_score,
                    'original_score': float(score),
                    'doc_id': idx,
                    'retrieval_query': query_text,
                    # No original query for docket files
                    **doc  # Include all document fields
                }
            
            search_result = SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                source="closed_search",
                published_date=published_date,
                metadata=metadata,
                result_id=str(idx)
            )
            
            results.append((combined_score, search_result))
            
            # Stop when we have enough results
            if len(results) >= k:
                break
    
    # Debug: log filtering stats
    if initial_count > 0 and len(results) == 0:
        logger.warning(f"⚠️  All {initial_count} initial results were filtered out:")
        logger.warning(f"   - Date filter excluded: {date_filtered_count}")
        logger.warning(f"   - Null dates excluded: {null_date_excluded_count}")
        logger.warning(f"   - Field filters excluded: {field_filtered_count}")
        if date_filter:
            logger.warning(f"   - Date filter: {date_filter}")
        if field_filters:
            logger.warning(f"   - Field filters: {field_filters}")
        logger.warning(f"   - include_null_dates: {include_null_dates}")
    
    # Re-sort by combined scores and return top k SearchResult objects
    results.sort(key=lambda x: x[0], reverse=True)
    return [result[1] for result in results[:k]]


def search_qdrant_index(index_path, collection_name, query_text, model, model_name, k=10, use_readability_reranking=True, readability_bias=0.4, date_filter=None, field_filters=None, text_filters=None, must_have_fields=None, include_null_dates=True, allow_null_for_sources=None):
    """
    Search Qdrant index with a text query.
    
    Args:
        index_path: Path to the Qdrant index
        collection_name: Name of the Qdrant collection
        query_text: Text query
        model: Pre-loaded SentenceTransformer model
        model_name: Name of the model (for mixedbread-ai detection)
        k: Number of results to return
        use_readability_reranking: Whether to apply readability reranking
        readability_bias: Bias towards readability (0-1)
        allow_null_for_sources: List of source values for which null dates are allowed
    
    Returns:
        List of SearchResult objects with same format as BM25 (metadocument vs docket file logic)
    """
    # Initialize Qdrant client
    client = QdrantClient(path=index_path)
    
    # Generate query embedding using unified helper
    query_embedding = _encode_queries_for_model(model, model_name, [query_text], batch_size=8)
    
    # Normalize query embedding
    query_embedding = query_embedding.astype("float32")
    query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-12)
    
    # Create Qdrant filter from parameters
    query_filter = create_qdrant_filter(date_filter, field_filters, text_filters, must_have_fields, include_null_dates, allow_null_for_sources)
    
    # Calculate search limit (get more for post-filtering)
    search_limit = k * 3
    
    try:
        # Search Qdrant using query_points with exact search and optional filtering
        search_params = {
            "collection_name": collection_name,
            "query": query_embedding[0].tolist(),
            "limit": search_limit,  # Get more results for post-filtering
            "with_payload": True,
            "search_params": SearchParams(exact=True)  # Use exact search for precise results
        }
        
        # Add filter if present
        if query_filter:
            search_params["query_filter"] = query_filter
        
        search_result = client.query_points(**search_params)
        hits = search_result.points
        
        # Process results
        results = []
        for hit in hits:
            payload = hit.payload
            title = payload.get('title', 'N/A')
            contents = payload.get('contents', '')
            
            if use_readability_reranking:
                # Apply readability reranking
                combined_score = apply_readability_reranking(hit.score, contents, readability_bias)
            else:
                combined_score = hit.score
            
            # Determine search type based on index path (same logic as BM25)
            is_metadocument = "metadocuments" in index_path
            
            # Create SearchResult object based on document type (same logic as BM25)
            if is_metadocument:
                # Metadocument format
                title = payload.get('title', 'Untitled')
                url = payload.get('downloaded_link', payload.get('article_link', ''))
                snippet = payload.get('snippet', '')
                published_date = payload.get('published_date', '')
                
                # Create metadata with full payload data
                metadata = {
                    'score': combined_score,
                    'original_score': hit.score,
                    'point_id': hit.id,
                    'retrieval_query': query_text,
                    'original_query': payload.get('query'),
                    **payload  # Include all payload fields
                }
            else:
                # Docket file format
                title = payload.get('filename', 'Untitled')
                url = payload.get('filename', '')  # Use filename as URL for docket files
                snippet = payload.get('snippet', '')
                published_date = payload.get('published_date', '')
                
                # Create metadata with full payload data
                metadata = {
                    'score': combined_score,
                    'original_score': hit.score,
                    'point_id': hit.id,
                    'retrieval_query': query_text,
                    # No original query for docket files
                    **payload  # Include all payload fields
                }
            
            search_result = SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                source="closed_search",
                published_date=published_date,
                metadata=metadata,
                result_id=str(hit.id)
            )
            
            results.append((combined_score, search_result))
        
        # Sort by combined scores and return top k SearchResult objects
        results.sort(key=lambda x: x[0], reverse=True)
        return [result[1] for result in results[:k]]
        
    except Exception as e:
        print(f"Error searching collection {collection_name}: {e}")
        return []


# =============================================================================
# READABILITY FUNCTIONS
# =============================================================================

def calculate_readability_score(text: str) -> float:
    """
    Calculate a readability score using Flesch-Kincaid and Dale-Chall metrics.
    
    - Flesch-Kincaid: Designed for technical content, used by U.S. Army for technical manuals
    - Dale-Chall: Based on familiar words rather than syllable/letter counts, better for
      technical content where familiar terminology is crucial for understanding
    
    Args:
        text: Text to score
        
    Returns:
        float: Normalized readability score (0-1, higher = more readable)
    """
    try:
        # Need at least 100 words for both metrics
        if len(text.split()) < 100:
            return 0.5  # Neutral score for short texts
        
        r = Readability(text)
        scores = []
        
        # Flesch-Kincaid Grade Level: 0-20+ (lower = more readable)
        try:
            fk = r.flesch_kincaid()
            fk_score = max(0, 1 - fk.score / 20.0)  # Normalize: grade 0-20 maps to 1-0
            scores.append(fk_score)
        except:
            pass
            
        # Dale-Chall: 0-10 (lower = more readable)
        try:
            dc = r.dale_chall()
            dc_score = max(0, 1 - dc.score / 10.0)  # Normalize: 0-10 maps to 1-0
            scores.append(dc_score)
        except:
            pass
        
        # Return average of available scores, or neutral if none worked
        if scores:
            return sum(scores) / len(scores)
        else:
            return 0.5
            
    except Exception:
        # If readability calculation fails, return neutral score
        return 0.5


def apply_readability_reranking(score, contents, readability_bias=0.4):
    """
    Apply readability reranking to a similarity score.
    
    Args:
        score: Original similarity score
        contents: Document contents text
        readability_bias: Bias towards readability (0-1, higher = more readability bias)
    
    Returns:
        Combined score blending similarity and readability
    """
    # Calculate readability score
    readability_score = calculate_readability_score(contents)
    
    # Simplicity score from word count
    word_count = len(contents.split())
    if word_count <= 400:
        simplicity_score = 0.80
    elif word_count <= 800:
        simplicity_score = 0.60
    elif word_count <= 1500:
        simplicity_score = 0.40
    elif word_count <= 3000:
        simplicity_score = 0.20
    elif word_count <= 5000:
        simplicity_score = 0.10
    else:
        simplicity_score = 0.05

    # Composite readability metric
    composite_readability = 0.7 * readability_score + 0.3 * simplicity_score

    # Clamp readability_bias into [0,1]
    rb = max(0.0, min(1.0, readability_bias))

    # Final score blends semantic similarity with readability
    return (1.0 - rb) * score + rb * composite_readability


# =============================================================================
# HIGH-LEVEL ORCHESTRATION FUNCTIONS
# =============================================================================

def build_dense_index_for_collection(collection_path, index_path, model_name, 
                                   index_type="Flat", nlist=100, batch_size=32, 
                                   backend="faiss", overwrite=False, dimensions=None, collection_name=None, device=None):
    """
    Build dense index for a single collection.
    
    Args:
        collection_path: Path to JSONL file or directory of JSON files
        index_path: Path to save the index
        model_name: HuggingFace model name
        index_type: Type of Faiss index (only used for backend="faiss")
        nlist: Number of clusters for IVFFlat (only used for backend="faiss")
        batch_size: Batch size for embedding generation
        backend: Index backend ("faiss" or "qdrant")
        overwrite: Whether to overwrite existing indexes (default: False)
        dimensions: Number of dimensions for embeddings (supports MRL for mixedbread-ai models)
        collection_name: Collection name for Qdrant (only used for backend="qdrant")
    """
    print(f"Building {backend} dense index for {collection_path}")
    
    # Load documents
    documents = load_documents(collection_path)
    print(f"Loaded {len(documents)} documents")
    
    # Generate embeddings
    embeddings = generate_embeddings(documents, model_name, batch_size, dimensions, device)
    
    if backend == "faiss":
        # Build Faiss index
        index = build_faiss_index(embeddings, index_type, nlist)
        
        # Save index and metadata
        save_faiss_index(index, documents, index_path)
        
        print(f"Successfully built Faiss index for {len(documents)} documents")
        return index, documents
        
    elif backend == "qdrant":
        # Use provided collection name or derive from path
        if collection_name:
            coll_name = collection_name
        else:
            # Fallback: derive from collection path
            if os.path.isdir(collection_path):
                coll_name = os.path.basename(collection_path)
            else:
                coll_name = os.path.splitext(os.path.basename(collection_path))[0]
        
        # Build Qdrant index
        return build_qdrant_index(embeddings, documents, index_path, coll_name, overwrite)
        
    else:
        raise ValueError(f"Unsupported backend: {backend}. Must be 'faiss' or 'qdrant'")


def build_dense_index_for_collection_with_model(collection_path, index_path, model, model_name, 
                                   index_type="Flat", nlist=100, batch_size=32, 
                                   backend="faiss", overwrite=False, dimensions=None, collection_name=None,
                                   prepend_summary=False):
    """
    Build dense index for a single collection using a pre-loaded model.
    
    Args:
        collection_path: Path to JSONL file or directory of JSON files
        index_path: Path to save the index
        model: Pre-loaded SentenceTransformer model
        model_name: HuggingFace model name (for metadata/logging)
        index_type: Type of Faiss index (only used for backend="faiss")
        nlist: Number of clusters for IVFFlat (only used for backend="faiss")
        batch_size: Batch size for embedding generation
        backend: Index backend ("faiss" or "qdrant")
        overwrite: Whether to overwrite existing indexes (default: False)
        dimensions: Number of dimensions for embeddings (supports MRL for mixedbread-ai models)
        collection_name: Collection name for Qdrant (only used for backend="qdrant")
        prepend_summary: Whether to prepend 'summary' field to 'contents' before embedding (default: False)
    """
    print(f"Building {backend} dense index for {collection_path}")
    if prepend_summary:
        print("  -> Using summary-prepended contents for embedding")
    
    # Load documents
    documents = load_documents(collection_path)
    print(f"Loaded {len(documents)} documents")
    
    # Prepare documents (optionally prepend summary to contents)
    prepared_documents = prepare_documents_with_summary(documents, prepend_summary)
    
    # Generate embeddings using pre-loaded model (on prepared documents)
    embeddings = generate_embeddings_with_model(prepared_documents, model, model_name, batch_size, dimensions)
    
    if backend == "faiss":
        # Build Faiss index
        index = build_faiss_index(embeddings, index_type, nlist)
        
        # Save index and metadata
        save_faiss_index(index, documents, index_path)
        
        print(f"Successfully built Faiss index for {len(documents)} documents")
        return index, documents
        
    elif backend == "qdrant":
        # Use provided collection name or derive from path
        if collection_name:
            coll_name = collection_name
        else:
            # Fallback: derive from collection path
            if os.path.isdir(collection_path):
                coll_name = os.path.basename(collection_path)
            else:
                coll_name = os.path.splitext(os.path.basename(collection_path))[0]
        
        # Build Qdrant index
        return build_qdrant_index(embeddings, documents, index_path, coll_name, overwrite)
        
    else:
        raise ValueError(f"Unsupported backend: {backend}. Must be 'faiss' or 'qdrant'")


def prepare_documents_with_summary(documents, prepend_summary=False, prefix_metadata=False):
    """
    Prepare documents for embedding, optionally prepending summary or metadata to contents.
    
    This leverages the RoPE positional encoding of long-context models which
    focuses attention on earlier tokens. By prepending metadata or summary, the embedding
    better captures what the document actually is.
    
    Args:
        documents: List of document dicts with 'contents' and optionally 'summary' fields
        prepend_summary: If True, prepend summary to contents
        prefix_metadata: If True, prefix structured metadata (link_name, proceeding_title) to contents
        
    Returns:
        List of modified document dicts (copies, originals unchanged)
    """
    if not prepend_summary and not prefix_metadata:
        return documents
    
    modified_docs = []
    summary_count = 0
    metadata_count = 0
    
    for doc in documents:
        modified_doc = doc.copy()
        contents = doc.get('contents', '')
        
        # Build prefix parts
        prefix_parts = []
        
        if prefix_metadata:
            # Explicitly prefix structured metadata to make document type prominent
            # This helps distinguish "Main Document" from "Certificate of Service" etc.
            metadata_parts = []
            
            # Add link_name (most important for distinguishing document types)
            link_name = doc.get('link_name', '')
            if link_name:
                metadata_parts.append(f"Document Type: {link_name}")
                metadata_count += 1
            
            # Add proceeding_title (helps identify which brief/document)
            proceeding_title = doc.get('proceeding_title', '')
            if proceeding_title:
                # Truncate if very long
                if len(proceeding_title) > 200:
                    proceeding_title = proceeding_title[:200] + "..."
                metadata_parts.append(f"Legal Proceeding: {proceeding_title}")
            
            # Add filename if available (can help with party names)
            filename = doc.get('filename', '')
            if filename and not link_name:  # Only if link_name not available
                # Extract meaningful part (remove timestamps)
                filename_clean = filename.split('_', 1)[-1] if '_' in filename else filename
                if len(filename_clean) > 150:
                    filename_clean = filename_clean[:150] + "..."
                metadata_parts.append(f"Document: {filename_clean}")
            
            if metadata_parts:
                prefix_parts.append("[DOCUMENT METADATA]\n" + "\n".join(metadata_parts))
        
        if prepend_summary:
            summary = doc.get('summary', '')
            if summary:
                prefix_parts.append(f"[DOCUMENT SUMMARY]\n{summary}")
                summary_count += 1
        
        # Combine prefix with contents
        if prefix_parts:
            prefix = "\n\n".join(prefix_parts) + "\n\n[FULL CONTENT]\n"
            modified_doc['contents'] = prefix + contents
        else:
            modified_doc['contents'] = contents
        
        modified_docs.append(modified_doc)
    
    if prepend_summary:
        print(f"Prepended summary to {summary_count}/{len(documents)} documents")
    if prefix_metadata:
        print(f"Prefixed metadata to {metadata_count}/{len(documents)} documents")
    
    return modified_docs


def build_dense_indexes(collection_paths, index_paths, model_name, 
                       index_type="Flat", nlist=100, batch_size=32, overwrite=False, 
                       backend="faiss", dimensions=None, collection_names=None, device=None,
                       prepend_summary=False):
    """
    Build dense indexes for multiple collections.
    
    Args:
        collection_paths: List of paths to JSONL files or directories of JSON files
        index_paths: List of paths to save indexes
        model_name: HuggingFace model name
        index_type: Type of Faiss index (only used for backend="faiss")
        nlist: Number of clusters for IVFFlat (only used for backend="faiss")
        batch_size: Batch size for embedding generation
        overwrite: Whether to overwrite existing indexes (default: False)
        backend: Index backend ("faiss" or "qdrant")
        dimensions: Number of dimensions for embeddings (supports MRL for mixedbread-ai models)
        collection_names: List of collection names for Qdrant (only used for backend="qdrant")
        prepend_summary: Whether to prepend 'summary' field to 'contents' before embedding (default: False)
    """
    # Filter paths that don't exist or don't have write permissions
    filtered_collection_paths = []
    filtered_index_paths = []
    
    for collection_path, index_path in zip(collection_paths, index_paths):
        # Check if index already exists (unless overwrite is True)
        if backend == "faiss":
            index_exists = os.path.exists(os.path.join(index_path, "faiss_index.bin"))
        else:  # qdrant
            # For Qdrant, check if the collection directory exists
            if os.path.isdir(collection_path):
                collection_name_from_path = os.path.basename(collection_path)
            else:
                collection_name_from_path = os.path.splitext(os.path.basename(collection_path))[0]
            qdrant_path = os.path.join(index_path, collection_name_from_path)
            index_exists = os.path.exists(qdrant_path)
        
        if not overwrite and index_exists:
            print(f"Skipping {collection_path} - index already exists")
            continue
        elif overwrite and index_exists:
            print(f"Overwrite mode: will rebuild index for {collection_path}")
            
        # Check read permissions
        try:
            if os.access(collection_path, os.R_OK):
                filtered_collection_paths.append(collection_path)
                filtered_index_paths.append(index_path)
            else:
                print(f"Skipping {collection_path} - no read permissions")
                continue
        except (OSError, FileNotFoundError):
            print(f"Error checking permissions of {collection_path}")
            continue
    
    print(f"Filtered collection paths: {filtered_collection_paths}")
    print(f"Total collections to process: {len(filtered_collection_paths)}")
    
    if not filtered_collection_paths:
        print("No collections to process")
        return
    
    # Load model once for all collections
    print(f"Loading model '{model_name}' once for all {len(filtered_collection_paths)} collections...")
    model = load_embedding_model(model_name, device=device)
    print("Model loaded successfully!")
    
    # Process collections sequentially with shared model
    for i, (collection_path, index_path) in enumerate(tqdm(zip(filtered_collection_paths, filtered_index_paths), 
                                          total=len(filtered_collection_paths), 
                                          desc="Processing collections")):
        try:
            collection_name = collection_names[i] if collection_names else None
            build_dense_index_for_collection_with_model(
                collection_path, index_path, model, model_name, index_type, nlist, batch_size,
                backend=backend, overwrite=overwrite, dimensions=dimensions, collection_name=collection_name,
                prepend_summary=prepend_summary
            )
            print(f"Successfully processed {collection_path}")
        except Exception as e:
            print(f"Error processing {collection_path}: {e}")
            continue
