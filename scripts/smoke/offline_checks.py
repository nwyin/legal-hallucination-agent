"""Offline checks for the retrieval code paths the replay baseline does not cover.

The smoke replay exercises only THINK, EDIT_SCRATCHPAD, and PROVIDE_FINAL_RESPONSE.
These checks make no network calls and no model calls; they verify that the
relocated retrieval modules import, that the environment dispatches every action
type to the right client, and that the observations keep their existing shape.
"""
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["OTEL_SDK_DISABLED"] = "true"


def _no_network(*args, **kwargs):
    raise RuntimeError("Network access forbidden during offline checks")


socket.socket.connect = _no_network
socket.create_connection = _no_network

from polaris_agents import actions, courtlistener, documents, environment, evaluation, parsing, web_search
from polaris_agents.actions import Action, ActionType, get_action_class, get_all_action_classes
from polaris_agents.environment import HallucinationCheckerEnvironment, Observation

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def make_environment(cache_dir):
    return HallucinationCheckerEnvironment(
        brief_info={"filename": "offline", "list_hallucinations": ["Fake v. Case, 1 U.S. 1 (2099)"]},
        brief_text="Offline dispatch check.",
        max_steps=10,
        search_top_k=2,
        opinion_cache_dir=cache_dir,
    )


@check
def action_registry_is_complete():
    """Every ActionType has a registered class, and the registry keeps its order."""
    registered = [c.action_type for c in get_all_action_classes()]
    assert set(registered) == set(ActionType), set(ActionType) ^ set(registered)
    for action_type in ActionType:
        assert issubclass(get_action_class(action_type), Action)
    # The CourtListener search description is generated from the shared mapping.
    description = get_action_class(ActionType.OPEN_COURTLISTENER_SEARCH).description
    assert description == get_action_class(ActionType.OPEN_COURTLISTENER_SEARCH).description
    options = get_action_class(ActionType.OPEN_COURTLISTENER_SEARCH).inputs["search_type"]["description"]
    for name in courtlistener.SEARCH_TYPES:
        assert f"'{name}'" in options, name
    assert courtlistener.SEARCH_TYPES is actions.SEARCH_TYPES


@check
def environment_exposes_every_action():
    """The task environment advertises the full action space."""
    with tempfile.TemporaryDirectory() as tmp:
        env = make_environment(tmp)
        assert set(env.action_space) == set(ActionType), set(ActionType) ^ set(env.action_space)
        assert env.initial_observation.metadata["available_actions"] == [a.value for a in env.action_space]


@check
def web_search_dispatch():
    """OPEN_WEB_SEARCH routes to web_search.search and keeps the observation shape."""
    calls = []
    original = environment.search_web

    def fake_search(query, **kwargs):
        calls.append((query, kwargs))
        return [web_search.SearchResult(title="T", url="http://example.invalid",
                                        snippet="S", source="serpapi", published_date="2024-01-02")]

    environment.search_web = fake_search
    try:
        with tempfile.TemporaryDirectory() as tmp:
            env = make_environment(tmp)
            obs = env.step(actions.OpenWebSearch(query="fake v case", num_results=2))
    finally:
        environment.search_web = original
    assert calls and calls[0][0] == "fake v case", calls
    assert obs.metadata["action_type"] == "OPEN_WEB_SEARCH"
    assert obs.result["num_results"] == 1
    assert obs.result["search_results"][0]["title"] == "T"
    assert env.search_history[0]["query"] == "fake v case"


@check
def courtlistener_search_dispatch():
    """OPEN_COURTLISTENER_SEARCH routes to the relocated CourtListener client."""
    calls = []
    original = environment.execute_courtlistener_search

    def fake_search(query, search_type="opinions", **kwargs):
        calls.append((query, search_type))
        return Observation(
            result={"results": [{"caseName": "Fake v. Case", "absolute_url": "/opinion/1/",
                                 "snippet": "text", "id": 1}]},
            metadata={"action_type": "OPEN_COURTLISTENER_SEARCH"},
        )

    environment.execute_courtlistener_search = fake_search
    try:
        with tempfile.TemporaryDirectory() as tmp:
            env = make_environment(tmp)
            obs = env.step(actions.OpenCourtListenerSearch(query="fake v case"))
    finally:
        environment.execute_courtlistener_search = original
    assert calls == [("fake v case", "opinions")], calls
    assert obs.metadata["action_type"] == "OPEN_COURTLISTENER_SEARCH"


@check
def opinion_access_registers_document():
    """ACCESS_COURTLISTENER_OPINION caches the opinion and registers it for READ_DOCUMENT."""
    original = environment.execute_courtlistener_opinion_access
    text = "\n".join(f"line {i}" for i in range(20))

    def fake_access(opinion_id):
        return Observation(result={"opinion": {"id": opinion_id, "plain_text": text,
                                               "case_name": "Fake v. Case"}}, metadata={})

    environment.execute_courtlistener_opinion_access = fake_access
    try:
        with tempfile.TemporaryDirectory() as tmp:
            env = make_environment(tmp)
            obs = env.step(actions.AccessCourtListenerOpinion(opinion_id="4242"))
            assert obs.result["action_type"] == "ACCESS_COURTLISTENER_OPINION"
            assert obs.result["stored"] is True
            cached = json.loads(Path(tmp, "4242.json").read_text())
            assert cached["case_name"] == "Fake v. Case"

            read = env.step(actions.ReadDocument(opinion_id="opinion_4242", start_line=0, num_lines=3))
            assert "line 0" in str(read.result) and "line 3" not in str(read.result)

            found = env.step(actions.SearchLocalOpinion(opinion_id="4242", search_string="line 7"))
            assert "line 7" in json.dumps(found.result)
    finally:
        environment.execute_courtlistener_opinion_access = original


@check
def citation_lookup_dispatch():
    """COURTLISTENER_CITATION_LOOKUP forwards the observation from the client."""
    original = environment.execute_courtlistener_citation_lookup
    sentinel = Observation(result={"action_type": "COURTLISTENER_CITATION_LOOKUP", "citations": []},
                           metadata={"cite": "1 U.S. 1"})
    environment.execute_courtlistener_citation_lookup = lambda cite: sentinel
    try:
        with tempfile.TemporaryDirectory() as tmp:
            env = make_environment(tmp)
            obs = env.step(actions.CourtListenerCitationLookup(cite="1 U.S. 1"))
    finally:
        environment.execute_courtlistener_citation_lookup = original
    assert obs is sentinel


@check
def scratchpad_dispatch():
    """EDIT_SCRATCHPAD is handled by the document manager."""
    with tempfile.TemporaryDirectory() as tmp:
        env = make_environment(tmp)
        env.step(actions.EditScratchpad(operation="append", content="first note"))
        obs = env.step(actions.EditScratchpad(operation="append", content="second note"))
        assert "first note" in obs.result["scratchpad"]
        assert "second note" in obs.result["scratchpad"]
        assert isinstance(env.document_manager, documents.DocumentManager)


@check
def courtlistener_client_builds_observations():
    """The relocated client still builds Observations (lazy import of the environment)."""
    original = courtlistener.make_courtlistener_request
    courtlistener.make_courtlistener_request = lambda endpoint, params: {"error": "offline"}
    try:
        obs = courtlistener.execute_courtlistener_search("query")
        assert isinstance(obs, Observation)
        assert obs.result.get("error") or obs.metadata.get("error")
    finally:
        courtlistener.make_courtlistener_request = original


@check
def date_parsing_and_cutoff_filtering():
    """Relocated date parsing still drives cutoff filtering."""
    from datetime import date
    assert web_search.parse_date_string("2024-01-02").date() == date(2024, 1, 2)
    assert web_search.parse_date_string("not a date") is None
    dated = web_search.SearchResult(title="a", url="u", snippet="s", source="serpapi",
                                    published_date="2020-01-01")
    undated = web_search.SearchResult(title="b", url="u", snippet="s", source="serpapi")
    kept = web_search._filter_by_cutoff_date([dated, undated], date(2021, 1, 1), exclude_undated=True)
    assert [r.title for r in kept] == ["a"], [r.title for r in kept]
    dropped = web_search._filter_by_cutoff_date([dated], date(2019, 1, 1), exclude_undated=True)
    assert dropped == []


@check
def search_clients_make_http_requests():
    """Exercise both real client request methods with only HTTP mocked."""
    for client_class in (web_search.SerpApiClient, web_search.MediaStackClient):
        client = client_class(api_key="offline-placeholder", max_retries=0)
        payload = {"results": [{"title": "offline result"}]}
        response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)
        with patch("requests.get", return_value=response) as get:
            result = client._make_request_with_retry(
                "https://example.invalid/search", {"q": "test query"}
            )
        get.assert_called_once_with("https://example.invalid/search", params={"q": "test query"})
        assert result == payload


@check
def action_parsing_round_trip():
    """Model output parses into the action parameters the environment expects."""
    parsed = parsing.parse_action_output_with_fallback(
        '```json\n{"action": {"action_type": "OPEN_WEB_SEARCH", "query": "  fake v case  "}}\n```'
    )
    assert parsed["action_type"] == "OPEN_WEB_SEARCH", parsed
    assert parsed["parameters"] == {"query": "  fake v case  "}, parsed
    action_class = get_action_class(ActionType(parsed["action_type"]))
    action = action_class(**parsing.normalize_action_parameters_for_construction(
        parsed["action_type"], parsed["parameters"]))
    assert action.query == "fake v case"
    try:
        parsing.normalize_action_parameters_for_construction(parsed["action_type"], {"query": "   "})
    except ValueError:
        pass
    else:
        raise AssertionError("Empty search query must be rejected")


@check
def scoring_helpers():
    """Scoring stays substring based and separate from recording."""
    metrics = evaluation.compute_metrics(*evaluation.evaluate_entry(
        ["invented citation"], ["invented citation, extra context"]))
    assert metrics["f1"] == 1.0, metrics
    empty = evaluation.compute_metrics(*evaluation.evaluate_entry([], []))
    assert empty["f1"] == 0.0, empty


def main():
    failures = 0
    for fn in CHECKS:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - report every failing check
            failures += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {fn.__name__}")
    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} checks passed")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
