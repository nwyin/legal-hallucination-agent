"""
Legal Hallucination Checker Environment Prompts and Domain Knowledge.

This module contains:
1. Domain knowledge provider (θ and D definitions) for legal hallucination checker
2. Environment-specific prompts for the legal hallucination checker task
3. Classification guidance for the action classifier

Combines domain understanding with task-specific prompt templates.
"""

from typing import Dict, Any, Optional

from ..base import DomainKnowledgeProvider


# ============================================================================
# Domain Knowledge Provider
# ============================================================================

class LegalHallucinationCheckerDomainKnowledge(DomainKnowledgeProvider):
    """
    Domain knowledge for legal citation hallucination checker.
    
    θ represents task instance-specific information about which citations/sentences are hallucinated:
    - For each citation in the brief: whether it exists, is accurately quoted, has correct pincites, supports the correct holding
    - Evidence from search results and opinion fetches that confirms or refutes each citation
    """
    
    def get_theta_description(self) -> str:
        return """### Task Parameters (θ)
Task instance-specific information needed to identify hallucinated citations in the brief.

Information that directly updates your knowledge about which citations/sentences are hallucinated:
- For each citation: whether the case exists, whether the case name matches the case name in the citation, whether the quoted language appears in the opinion, whether the pincite is correct
- Evidence from CourtListener searches and opinion fetches that confirms or refutes each citation
- Any misattributions, fabricated quotes, or non-existent case citations you discover

### Domain knowledge: legal case citations
- **Citation format**: 
    Typical format is Case Name, Volume Reporter Page (e.g., 557 F.2d 170). The part after the comma is the reporter citation; "at" introduces a pincite (specific page).
- **Quotation format**:
    ONLY words, phrases, and sentences that are inside quotation marks are considered direct quotes. 
    (i.e. in sentence: The PLRA never takes more than 20 percent of a prisoner's assets not only supports the Courts holding of constitutionality, but was a "critical factor" in it, the quoted text is ONLY critical factor, which should appear in the original opinion, without quotes.)
    The presence of ellipses (...) or bracketed ellipses ([...]) only permits omission of intervening text.
    (i.e. in sentence: [W]here Congress explicitly enumerates certain exceptions to a general prohibition), the word [W]here would NOT appear verbatim in the opinion and should be OMITTED from string match searches.)
    Return the entire sentence as hallucinated if any visible portion of the quotation does not appear verbatim in the opinion. 
- **Holding format**: The holding the citation is usually IMMEDIATELY PRECEDES the citation.

If the case does not exist or the name does not match, the entire citation is hallucinated—no need to separately verify quotes or holdings for that citation.
- **Verification hierarchy**: 
    If a citation is already determined to be hallucinated (e.g., case does not exist or case name does not match), treat all quotes and holdings under that citation as hallucinated.
    In this case, ONLY return the citations that are hallucinated, no need to return the quotes and holdings."""

    def get_domain_knowledge_description(self) -> str:
        return """### Domain knowledge: legal case citations
- **Citation format**:
    Typical format is Case Name, Volume Reporter Page (e.g., 557 F.2d 170). The part after the comma is the reporter citation; "at" introduces a pincite (specific page).
- **Quotation format**:
    ONLY words, phrases, and sentences that are inside quotation marks are considered direct quotes.
    (i.e. in sentence: The PLRA never takes more than 20 percent of a prisoner's assets not only supports the Courts holding of constitutionality, but was a "critical factor" in it, the quoted text is ONLY critical factor, which should appear in the original opinion, without quotes.)
    The presence of ellipses (...) or bracketed ellipses ([...]) only permits omission of intervening text.
    (i.e. in sentence: [W]here Congress explicitly enumerates certain exceptions to a general prohibition), the word [W]here would NOT appear verbatim in the opinion and should be OMITTED from string match searches.)
    Return the entire sentence as hallucinated if any visible portion of the quotation does not appear verbatim in the opinion.
- **Holding format**: The holding the citation is usually IMMEDIATELY PRECEDES the citation.

If the case does not exist or the name does not match, the entire citation is hallucinated—no need to separately verify quotes or holdings for that citation.
- **Verification hierarchy**:
    If a citation is already determined to be hallucinated (e.g., case does not exist or case name does not match), treat all quotes and holdings under that citation as hallucinated.
    In this case, ONLY return the citations that are hallucinated, no need to return the quotes and holdings."""

    def get_design_description(self) -> str:
        return """### Design / domain knowledge
How to verify legal citations and detect hallucinations: 
- BlueBook citation format, CourtListener search and opinion fetch, common hallucination types (non-existent cases, misquoted language, wrong pincites)."""

    def get_classification_guidance(self) -> str:
        return """Score each action 0.0–1.0 by how directly it gathers evidence for identifying hallucinated citations. Search and opinion actions: 0.7–0.95. THINK: 0.2–0.5."""

    def get_action_selection_task_section(self) -> Optional[str]:
        return """You are verifying citations in a legal brief for hallucinations. Your uncertainty (θ) is: which citations are hallucinated (non-existent, misquoted, or wrong pincite). You reduce that uncertainty by using search and opinion actions to gather evidence; then you submit your final list of hallucinated citations."""

    def get_action_selection_guidance(self) -> Optional[str]:
        return """For this task you must **verify citations** by gathering evidence from external sources. Apply the following when choosing actions:

- **THINK has zero information gain**: The observation from THINK only echoes your thought
- **To reduce uncertainty about θ, you must use**: OPEN_COURTLISTENER_SEARCH (find cases), OPEN_WEB_SEARCH (existence checks), ACCESS_COURTLISTENER_OPINION (fetch opinion by ID), SEARCH_LOCAL_OPINION (search within a fetched opinion for quoted language). These actions return new evidence; THINK does not.
- **Reserve PROVIDE_FINAL_RESPONSE** until you have used search/opinion actions to verify ALL citations, or you have exhausted steps."""


# ============================================================================
# Environment Prompts
# ============================================================================

def get_search_capabilities_open_search() -> str:
    """
    Get search capabilities for hallucination checker (OPEN_WEB_SEARCH + CourtListener).
    """
    return """## Search Capabilities

You have access to:

- **COURTLISTENER_CITATION_LOOKUP**: Resolve a reporter citation (e.g., `934 F.3d 53`, `143 S. Ct. 1196`) to canonical case info. Use `cite`.
- **OPEN_COURTLISTENER_SEARCH**: Search CourtListener by case name or citation text. Use `query`.
- **ACCESS_COURTLISTENER_OPINION**: Fetch an opinion by `opinion_id`. Full text is stored locally and registered for reading.
- **SEARCH_LOCAL_OPINION**: Search within a fetched opinion using `opinion_id` and `search_string`.
- **READ_DOCUMENT**: Read a fetched opinion in sections. Use `opinion_id` = `opinion_<id>` (or just the numeric `<id>`), `start_line` (0-indexed), and `num_lines`. Use this to read the full opinion in chunks when SEARCH_LOCAL_OPINION is not enough.
- **EDIT_SCRATCHPAD**: Take notes. Use `operation`: `append` (add to end), `insert` (at `position`), `replace` (at `position`), or `clear`. Use `content` for the text and `position` (0-indexed) for insert/replace.
- **OPEN_WEB_SEARCH**: Search the open web for legal information. Use `query`.

---
### Case-law search policy (important)
- If a **reporter-style citation** is present, **use COURTLISTENER_CITATION_LOOKUP first**.
- Use OPEN_COURTLISTENER_SEARCH only if citation lookup fails or only a case name is available.
- Use OPEN_WEB_SEARCH: CourtListener is inconclusive or fails to find the case.

---
### COURTLISTENER_CITATION_LOOKUP – How to use
- Input only the reporter citation AS IS from the brief.
- DO NOT omit 'at' and page numbers following it.
- If a match is returned, use the associated `opinion_id`.

---
### OPEN_COURTLISTENER_SEARCH – How to search
- Never include pincites or page numbers following “at” (they are pinpoint pages, not identifiers).
- Reduce citations to volume + reporter + first page only.
- Prefer **quoted reporter citations** (e.g., `"557 F.2d 170"`) or **case name queries**, not both.
- Do not combine multiple identifiers in one query.
- Expect reporter and punctuation variation (e.g., `F.2d` vs `F2d`).
- If no high-confidence match is found, escalate to OPEN_WEB_SEARCH.

---
### SEARCH_LOCAL_OPINION – How to search
- Search for SHORT distinctive substrings and use the returned snippet to verify the quoted language.
- Look for key noun phrases.
- DO NOT include words that are not in quotation marks and words in square brackets.
- Use READ_DOCUMENT to read the full opinion in sections if string search is inconclusive.

---
### READ_DOCUMENT – How to read the opinion
- Use the opinion_id to identify which opinion to read.
- Iteratively read the opinion in sections to find the quoted language or holdings.

---
### OPEN_WEB_SEARCH – How to search
- Do not issue strict multi-field queries (case name + docket + WL citation).
- Use loose, normalized case names without punctuation or “v.” formatting.
- Prefer either case name or reporter citation initially.
- Avoid quoting long strings unless retrying after a loose search fails.
- Use web search for existence checks, context, or recovery when CourtListener fails.

---
### General verification flow
1. Resolve the case (citation lookup → CourtListener search → web fallback).
2. Fetch the opinion by `opinion_id`.
3. Verify quoted language with SEARCH_LOCAL_OPINION.
4. Verify holdings with READ_DOCUMENT, and EDIT_SCRATCHPAD.
"""


def get_response_requirements() -> str:
    """
    Get task-specific response requirements for hallucination checker predictions.
    
    Returns:
        Response requirements string (list of hallucinated citations)
    """
    return (
        "Provide a **list** of hallucinated citations, case names,quotes, and holdings.\n\n"
        "**Format:** A JSON array of only the hallucinated segments, e.g. `[\"string1\", \"string2\"]`.\n\n"
        "**Completeness (required):** Your list must include **every** citation, quote, or holding that you have labeled as hallucinated in your Current Task Beliefs. "
        "Do not omit any segment you believe is hallucinated — the response is evaluated against the full set. If you have N items marked hallucinated in your beliefs, your response must contain exactly those N segments (or the citation alone when sub-items are implied).\n\n"
        "**Important**: If a citation itself is hallucinated, it is assumed that all the quotes and holdings within that citation are hallucinated as well so there is no need to return them separately. "
        "If no hallucinations are found, return an empty list: `[]`.\n\n"
        "**No reasoning in the final answer:** The `response` field must contain **only** the list —- no reasoning, explanation, or prose. Put all reasoning, analysis, and explanation in the `reasoning` field, not in `response`."
        " Only return the hallucinated segments: If a holding is hallucinated, return the FULL sentence of the holding."
    )

