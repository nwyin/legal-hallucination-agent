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

from benchmark_agent import actions, courtlistener, documents, environment, evaluation, parsing, web_search
from benchmark_agent.actions import Action, ActionType, get_action_class, get_all_action_classes
from benchmark_agent.environment import HallucinationCheckerEnvironment, Observation

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
    """OPEN_COURTLISTENER_SEARCH routes to the CourtListener client and wraps its summary."""
    calls = []
    original = environment.search_courtlistener

    def fake_search(query, search_type="opinions", **kwargs):
        calls.append((query, search_type))
        return {"count": 1, "api_type": "o", "results": [{"id": 1, "case_name": "Fake v. Case",
                "url": "/opinion/1/", "snippet": "text", "metadata": {}}]}

    environment.search_courtlistener = fake_search
    try:
        with tempfile.TemporaryDirectory() as tmp:
            env = make_environment(tmp)
            obs = env.step(actions.OpenCourtListenerSearch(query="fake v case"))
    finally:
        environment.search_courtlistener = original
    assert calls == [("fake v case", "opinions")], calls
    assert obs.metadata["action_type"] == "OPEN_COURTLISTENER_SEARCH"
    assert obs.result["search_results"][0]["title"] == "Fake v. Case"
    assert obs.metadata["raw_metadata"]["total_results"] == 1


@check
def opinion_access_registers_document():
    """ACCESS_COURTLISTENER_OPINION caches the opinion and registers it for READ_DOCUMENT."""
    original = environment.fetch_opinion
    text = "\n".join(f"line {i}" for i in range(20))
    environment.fetch_opinion = lambda opinion_id: {"id": opinion_id, "plain_text": text,
                                                    "case_name": "Fake v. Case"}
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
        environment.fetch_opinion = original


@check
def citation_lookup_dispatch():
    """COURTLISTENER_CITATION_LOOKUP wraps the client's match list in an observation."""
    original = environment.lookup_citation
    environment.lookup_citation = lambda cite: [{"citation": cite, "clusters": []}]
    try:
        with tempfile.TemporaryDirectory() as tmp:
            env = make_environment(tmp)
            obs = env.step(actions.CourtListenerCitationLookup(cite="1 U.S. 1"))
    finally:
        environment.lookup_citation = original
    assert obs.result["action_type"] == "COURTLISTENER_CITATION_LOOKUP"
    assert obs.result["citations"] == ["1 U.S. 1"], obs.result
    assert obs.metadata == {"action_type": "COURTLISTENER_CITATION_LOOKUP", "cite": "1 U.S. 1"}


@check
def scratchpad_dispatch():
    """EDIT_SCRATCHPAD is handled by the document manager."""
    with tempfile.TemporaryDirectory() as tmp:
        env = make_environment(tmp)
        env.step(actions.EditScratchpad(operation="append", content="first note"))
        obs = env.step(actions.EditScratchpad(operation="append", content="second note"))
        assert obs.result["scratchpad"] == "first note\nsecond note", obs.result
        env.step(actions.EditScratchpad(operation="insert", content="zeroth", position=0))
        obs = env.step(actions.EditScratchpad(operation="replace", content="2nd", position=2))
        assert obs.result["scratchpad"] == "zeroth\nfirst note\n2nd", obs.result
        obs = env.step(actions.EditScratchpad(operation="replace", content="x", position=9))
        assert obs.result["error"] == "edit failed", obs.result
        assert isinstance(env.document_manager, documents.DocumentManager)


@check
def courtlistener_errors_become_observations():
    """Client errors raise; the environment turns them into error observations for the agent."""
    original = courtlistener.make_courtlistener_request

    def offline(endpoint, params):
        raise courtlistener.NonRetryableError("400 - bad query")

    courtlistener.make_courtlistener_request = offline
    try:
        with tempfile.TemporaryDirectory() as tmp:
            env = make_environment(tmp)
            obs = env.step(actions.OpenCourtListenerSearch(query="bad query"))
            assert obs.metadata["non_retryable"] is True, obs.metadata
            assert "Please fix the query" in obs.result, obs.result
            obs = env.step(actions.AccessCourtListenerOpinion(opinion_id="1"))
            assert "bad query" in obs.metadata["error"], obs.metadata
    finally:
        courtlistener.make_courtlistener_request = original
    try:
        courtlistener.search_courtlistener("query", search_type="not-a-type")
    except ValueError:
        pass
    else:
        raise AssertionError("Unsupported search types must be rejected")


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
def search_client_makes_http_requests():
    """Exercise the real SerpAPI request method with only HTTP mocked."""
    client = web_search.SerpApiClient(api_key="offline-placeholder", max_retries=0)
    payload = {"results": [{"title": "offline result"}]}
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)
    with patch("requests.get", return_value=response) as get:
        result = client._make_request_with_retry("https://example.invalid/search", {"q": "test query"})
    get.assert_called_once_with("https://example.invalid/search", params={"q": "test query"})
    assert result == payload


@check
def web_search_failure_is_an_error_observation():
    """A missing SerpAPI key surfaces as an error, not as an empty result list."""
    with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"SERPAPI_API_KEY": ""}):
        env = make_environment(tmp)
        obs = env.step(actions.OpenWebSearch(query="fake v case"))
    assert obs.metadata.get("error"), obs.metadata
    assert "SERPAPI_API_KEY" in obs.metadata["error"]


@check
def web_search_keeps_undated_results():
    """Undated Google organic results reach the agent (the paper's web hits carry no dates)."""
    original = web_search.SerpApiClient.google_search
    web_search.SerpApiClient.google_search = lambda self, query, **kw: {
        "organic_results": [{"title": "undated", "link": "u", "snippet": "s", "position": 1}]}
    try:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"SERPAPI_API_KEY": "offline"}):
            env = make_environment(tmp)
            obs = env.step(actions.OpenWebSearch(query="fake v case"))
    finally:
        web_search.SerpApiClient.google_search = original
    assert obs.result["num_results"] == 1, obs.result
    assert obs.result["search_results"][0]["title"] == "undated"


@check
def action_parsing_round_trip():
    """Model output parses into the action parameters the environment expects."""
    action_type, parameters = parsing.parse_action_response(
        '```json\n{"action": {"action_type": "OPEN_WEB_SEARCH", "query": "  fake v case  "}}\n```'
    )
    assert action_type == "OPEN_WEB_SEARCH", action_type
    assert parameters == {"query": "fake v case"}, parameters  # omitted optionals stay omitted
    action = get_action_class(ActionType(action_type))(
        **parsing.normalize_action_parameters_for_construction(action_type, parameters))
    assert action.query == "fake v case" and action.search_type == "web"
    for bad in ('{"action": {"action_type": "OPEN_WEB_SEARCH", "query": "   "}}',
                '{"action": {"action_type": "NOT_AN_ACTION"}}', "no json here"):
        try:
            parsing.parse_action_response(bad)
        except (ValueError, parsing.ValidationError):
            pass
        else:
            raise AssertionError(f"Must reject: {bad}")


@check
def action_selection_reasks_on_invalid_output():
    """An invalid action response is re-asked with the error; the retry is used."""
    from benchmark_agent.agent import BOEDCitationTrackerAgent
    responses = iter(['{"action": {"action_type": "OPEN_WEB_SEARCH", "query": ""}}',
                      '{"action": {"action_type": "THINK", "thought": "second try"}}'])
    calls = []

    def fake_model_api(model_id, prompt, **kwargs):
        calls.append(prompt)
        return next(responses)

    with tempfile.TemporaryDirectory() as tmp:
        agent = BOEDCitationTrackerAgent(environment=make_environment(tmp), model_api=fake_model_api,
                                         model_id="offline", open_web_search_enabled=True)
        action = agent._call_llm_for_action_selection([{"role": "user", "content": "choose"}])
    assert isinstance(action, actions.Think) and action.thought == "second try", action
    assert len(calls) == 2 and len(calls[1]) == 3, [len(c) for c in calls]
    assert calls[1][1]["role"] == "assistant" and "not a valid action" in calls[1][2]["content"]


@check
def action_selection_rejects_disabled_actions():
    """An action outside the agent's configured action space is re-asked, not executed."""
    from benchmark_agent.agent import BOEDCitationTrackerAgent
    responses = iter(['{"action": {"action_type": "OPEN_WEB_SEARCH", "query": "fake v case"}}',
                      '{"action": {"action_type": "THINK", "thought": "no web search"}}'])
    calls = []

    def fake_model_api(model_id, prompt, **kwargs):
        calls.append(prompt)
        return next(responses)

    with tempfile.TemporaryDirectory() as tmp:
        agent = BOEDCitationTrackerAgent(environment=make_environment(tmp), model_api=fake_model_api,
                                         model_id="offline", open_web_search_enabled=False)
        action = agent._call_llm_for_action_selection([{"role": "user", "content": "choose"}])
    assert isinstance(action, actions.Think), action
    assert len(calls) == 2 and "not available in this run" in calls[1][2]["content"], calls[1]


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
