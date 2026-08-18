import time

import pytest

from cook_ad.llm import client as llm_client
from cook_ad.llm import detect, textify

VOCAB = {
    "verbs": {"stall": 0, "pour": 1, "take": 2, "stir": 3},
    "nouns": {"kitchen": 0, "bowl": 1, "cereals": 2, "milk": 3},
}


class StubClient:
    """Records every message list it is handed and replays canned replies in order. Keeps the
    whole suite offline -- nothing here touches the network."""

    def __init__(self, replies, concurrency=1):
        self.replies = list(replies)
        self.calls = []
        self.concurrency = concurrency

    def complete(self, messages):
        self.calls.append([dict(m) for m in messages])
        return self.replies[len(self.calls) - 1] if len(self.calls) <= len(self.replies) else "No Anomaly"

    def complete_many(self, message_lists, concurrency=None):
        return [self.complete(m) for m in message_lists]

    n_would_request = 0


_ID_TO_VERB = {i: v for v, i in VOCAB["verbs"].items()}
_ID_TO_NOUN = {i: n for n, i in VOCAB["nouns"].items()}


class _Lex:
    def verb(self, v):
        return _ID_TO_VERB[int(v)]

    def noun(self, n):
        return _ID_TO_NOUN[int(n)]


def _steps(n=3):
    """n steps that all render DIFFERENTLY -- otherwise a 'future step not shown' assertion
    passes vacuously because two steps share their text."""
    verbs, nouns = [], []
    for i in range(n):
        verbs += [1 + (i % 3)] * (i + 1)
        nouns += [1 + (i % 3)] * (i + 1)
    steps = textify.steps_from_ids(verbs, nouns, _Lex())
    assert len({textify.render_step(s) for s in steps}) == len(steps)
    return steps


# ---- parser ---------------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["No Anomaly", "no anomaly", "  NO ANOMALY.  ", "No Anomaly!"])
def test_parses_no_anomaly(text):
    v = detect.parse_response(text, 0, VOCAB)
    assert (v.is_anomaly, v.error_type, v.parse_ok) == (False, None, True)


def test_parses_full_strict_form():
    v = detect.parse_response(
        "substitution Anomaly. Correct move would have been pour milk for 12 seconds", 2, VOCAB
    )
    assert v.step_index == 2
    assert (v.is_anomaly, v.error_type, v.parse_ok) == (True, "substitution", True)
    assert v.correction == ("pour", "milk", 12)


def test_parses_type_without_a_correction():
    v = detect.parse_response("repetition anomaly", 0, VOCAB)
    assert (v.is_anomaly, v.error_type, v.correction, v.parse_ok) == (True, "repetition", None, True)


def test_chatty_reply_is_rescued_but_flagged_unparsed():
    v = detect.parse_response("Sure! I'd say this is a transposition Anomaly here.", 0, VOCAB)
    assert (v.is_anomaly, v.error_type) == (True, "transposition")
    assert v.parse_ok is False


def test_out_of_vocabulary_correction_is_dropped_not_guessed():
    """An out-of-vocab correction cannot be compared to the pre-injection step without fuzzy
    matching, which would put a tunable threshold inside the metric."""
    v = detect.parse_response(
        "abandonment Anomaly. Correct move would have been frobnicate widget for 5 seconds", 0, VOCAB
    )
    assert (v.is_anomaly, v.error_type, v.correction) == (True, "abandonment", None)


@pytest.mark.parametrize("text", ["", "the person seems fine to me", "???"])
def test_unparseable_is_recorded_not_coerced_to_no_anomaly(text):
    """Silently reading garbage as the majority class would inflate apparent specificity for
    free, so parse_ok must record the failure."""
    v = detect.parse_response(text, 0, VOCAB)
    assert v.is_anomaly is False
    assert v.parse_ok is False


def test_unknown_type_name_is_not_accepted_as_a_type():
    v = detect.parse_response("hallucination Anomaly.", 0, VOCAB)
    assert v.error_type is None


# ---- incremental protocol -------------------------------------------------------------------

def test_incremental_sends_one_request_per_step():
    steps = _steps(3)
    stub = StubClient(["No Anomaly", "substitution Anomaly.", "No Anomaly"])
    verdicts = detect.run_trial(stub, "SYS", steps, VOCAB, protocol="incremental")

    assert len(stub.calls) == len(steps) == 3
    assert [v.step_index for v in verdicts] == [0, 1, 2]
    assert [v.is_anomaly for v in verdicts] == [False, True, False]

    # exactly two turns per request: system, then the step list so far
    assert [len(c) for c in stub.calls] == [2, 2, 2]
    assert all(c[0] == {"role": "system", "content": "SYS"} for c in stub.calls)
    assert all(c[1]["role"] == "user" for c in stub.calls)


def test_incremental_does_not_feed_back_the_models_own_answers():
    """Prefix-only: causality is what the comparison needs, self-feedback is not, and feeding a
    false alarm at step 2 into steps 3..7 makes a later miss indistinguishable from contamination
    by that earlier mistake."""
    steps = _steps(3)
    stub = StubClient(["substitution Anomaly.", "No Anomaly", "No Anomaly"])
    detect.run_trial(stub, "SYS", steps, VOCAB, protocol="incremental")

    for call in stub.calls:
        assert not any(m["role"] == "assistant" for m in call)
        assert "substitution Anomaly." not in " ".join(m["content"] for m in call)


def test_incremental_requests_are_independent_of_earlier_replies():
    """The property that makes the sweep parallelisable / batch-submittable: a request is a pure
    function of (trial, step index), so different replies must produce identical prompts."""
    steps = _steps(3)
    a = StubClient(["No Anomaly"] * 3)
    b = StubClient(["omission Anomaly.", "repetition Anomaly.", "No Anomaly"])
    detect.run_trial(a, "SYS", steps, VOCAB, protocol="incremental")
    detect.run_trial(b, "SYS", steps, VOCAB, protocol="incremental")
    assert a.calls == b.calls


def test_incremental_prompts_grow_by_pure_append():
    """Request i's user turn must be a strict prefix of request i+1's, or prefix caching (vLLM
    APC, hosted implicit caching) cannot reuse it."""
    steps = _steps(4)
    stub = StubClient(["No Anomaly"] * 4)
    detect.run_trial(stub, "SYS", steps, VOCAB, protocol="incremental")

    contents = [c[1]["content"] for c in stub.calls]
    for earlier, later in zip(contents, contents[1:]):
        assert later.startswith(earlier)
    # ...and the last line of each is the step being judged
    for i, content in enumerate(contents):
        assert content.splitlines()[-1].endswith(textify.render_step(steps[i]))
    assert len(contents[-1].splitlines()) == 4


def test_incremental_never_shows_a_future_step():
    """The one property the comparison actually depends on: latency in steps is only meaningful if
    the model could not have seen the anomaly before the step it is judging."""
    steps = _steps(3)
    stub = StubClient(["No Anomaly"] * 3)
    detect.run_trial(stub, "SYS", steps, VOCAB, protocol="incremental")
    for i, call in enumerate(stub.calls):
        seen = " ".join(m["content"] for m in call)
        for future in steps[i + 1:]:
            assert textify.render_step(future) not in seen


# ---- batch protocol -------------------------------------------------------------------------

def test_batch_sends_one_request_and_parses_numbered_lines():
    steps = _steps(3)
    stub = StubClient(["1. No Anomaly\n2. omission Anomaly.\n3. No Anomaly"])
    verdicts = detect.run_trial(stub, "SYS", steps, VOCAB, protocol="batch")

    assert len(stub.calls) == 1
    assert [v.is_anomaly for v in verdicts] == [False, True, False]
    assert verdicts[1].error_type == "omission"
    listing = stub.calls[0][-1]["content"]
    assert all(textify.render_step(s) in listing for s in steps)


def test_batch_missing_line_is_a_parse_failure_not_an_implicit_pass():
    steps = _steps(3)
    stub = StubClient(["1. No Anomaly\n3. No Anomaly"])
    verdicts = detect.run_trial(stub, "SYS", steps, VOCAB, protocol="batch")

    assert len(verdicts) == 3
    assert verdicts[1].parse_ok is False
    assert verdicts[1].is_anomaly is False
    assert verdicts[0].parse_ok and verdicts[2].parse_ok


def test_request_cost_matches_the_protocols():
    steps = _steps(4)
    assert detect.request_cost(steps, "incremental") == 4
    assert detect.request_cost(steps, "batch") == 1


def test_unknown_protocol_raises():
    with pytest.raises(ValueError, match="unknown protocol"):
        detect.run_trial(StubClient([]), "SYS", _steps(1), VOCAB, protocol="telepathy")


# ---- client budget / cache ------------------------------------------------------------------

def test_dry_run_makes_no_calls_and_counts_the_cost():
    c = llm_client.ChatClient(dry_run=True)
    for i in range(5):
        assert c.complete([{"role": "user", "content": f"step {i}"}]) == "No Anomaly"
    assert c.stats()["requests_sent"] == 0
    assert c.stats()["uncached_requests_needed"] == 5


def test_budget_guard_raises_rather_than_truncating():
    c = llm_client.ChatClient(dry_run=True, max_requests=2)
    c.complete([{"role": "user", "content": "a"}])
    c.complete([{"role": "user", "content": "b"}])
    with pytest.raises(llm_client.BudgetExceeded):
        c.complete([{"role": "user", "content": "c"}])


def test_cache_round_trips_and_does_not_consume_budget(tmp_path):
    c = llm_client.ChatClient(cache_dir=tmp_path, dry_run=True)
    messages = [{"role": "user", "content": "pour milk for 9 seconds"}]
    c._cache_put(c._cache_key(messages), "omission Anomaly.", messages)

    assert c.complete(messages) == "omission Anomaly."
    assert c.stats()["cache_hits"] == 1
    assert c.stats()["uncached_requests_needed"] == 0


REMOTE = "https://generativelanguage.googleapis.com/v1beta/openai"


def test_missing_api_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("COOK_AD_TEST_KEY", raising=False)
    c = llm_client.ChatClient(base_url=REMOTE, api_key_env="COOK_AD_TEST_KEY")
    with pytest.raises(llm_client.LLMError, match="COOK_AD_TEST_KEY"):
        c._api_key()


def test_blank_key_is_treated_as_missing(monkeypatch):
    """.env.example ships GEMINI_API_KEY= with no value, so an unfilled copy must fail loudly
    rather than send an empty bearer token."""
    monkeypatch.setenv("COOK_AD_TEST_KEY", "")
    with pytest.raises(llm_client.LLMError):
        llm_client.ChatClient(base_url=REMOTE, api_key_env="COOK_AD_TEST_KEY")._api_key()


# ---- env file -------------------------------------------------------------------------------

def test_env_file_loads_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("COOK_AD_TEST_KEY", raising=False)
    f = tmp_path / ".env"
    f.write_text(
        "# a comment\n"
        "\n"
        'COOK_AD_TEST_KEY = "abc123"\n'
        "export COOK_AD_TEST_OTHER=xyz\n"
        "not_a_pair\n"
    )
    assert llm_client.load_env_file(f) == 2
    assert llm_client.ChatClient(api_key_env="COOK_AD_TEST_KEY")._api_key() == "abc123"
    monkeypatch.delenv("COOK_AD_TEST_KEY", raising=False)
    monkeypatch.delenv("COOK_AD_TEST_OTHER", raising=False)


def test_existing_environment_wins_over_the_env_file(tmp_path, monkeypatch):
    """So an explicit `KEY=... python run_llm_eval.py` is never silently overridden by a stale
    file."""
    monkeypatch.setenv("COOK_AD_TEST_KEY", "from-environment")
    f = tmp_path / ".env"
    f.write_text("COOK_AD_TEST_KEY=from-file\n")

    assert llm_client.load_env_file(f) == 0
    assert llm_client.ChatClient(api_key_env="COOK_AD_TEST_KEY")._api_key() == "from-environment"
    assert llm_client.load_env_file(f, override=True) == 1
    assert llm_client.ChatClient(api_key_env="COOK_AD_TEST_KEY")._api_key() == "from-file"


def test_missing_env_file_is_not_an_error(tmp_path):
    """CI has no .env, and exporting the variable directly is equally valid."""
    assert llm_client.load_env_file(tmp_path / "nope.env") == 0


def test_remote_fallback_still_names_the_gemini_key():
    """Switching --base-url to a hosted provider should not also require naming the key variable."""
    assert llm_client.DEFAULT_API_KEY_ENV == "GEMINI_API_KEY"
    assert llm_client.REMOTE_RPM == 15


# ---- 429 / quota handling --------------------------------------------------------------------

def _http_error(code, body, headers=None):
    import io
    import urllib.error
    return urllib.error.HTTPError(
        url="http://x/chat/completions", code=code, msg="err",
        hdrs=headers or {}, fp=io.BytesIO(body.encode()),
    )


def test_retry_delay_read_from_retry_after_header():
    e = _http_error(429, "{}", headers={"Retry-After": "42"})
    assert llm_client._parse_retry_delay(e, "{}") == 42.0


def test_retry_delay_read_from_google_retryinfo_body():
    """The Gemini API puts the delay in the body, not the header. The header-only version silently
    discarded the one authoritative signal about how long to wait."""
    body = '{"error":{"code":429,"details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo",'\
           '"retryDelay":"57s"}]}}'
    assert llm_client._parse_retry_delay(_http_error(429, body), body) == 57.0


def test_retry_delay_none_when_server_says_nothing():
    assert llm_client._parse_retry_delay(_http_error(429, "not json"), "not json") is None
    assert llm_client._parse_retry_delay(_http_error(429, "{}"), "{}") is None


def test_http_date_retry_after_falls_through_instead_of_crashing():
    e = _http_error(429, "{}", headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})
    assert llm_client._parse_retry_delay(e, "{}") is None


@pytest.mark.parametrize("body", [
    '{"error":{"message":"Quota exceeded","details":[{"quotaId":'
    '"GenerateRequestsPerDayPerProjectPerModel"}]}}',
    '{"error":{"message":"free-models-per-day limit reached"}}',
    '{"error":{"message":"Daily limit exceeded"}}',
])
def test_per_day_quota_raises_immediately_without_burning_retries(monkeypatch, body):
    """Waiting cannot clear a daily quota, so retrying it 6 times just delays the crash."""
    monkeypatch.setenv("COOK_AD_TEST_KEY", "k")
    c = llm_client.ChatClient(api_key_env="COOK_AD_TEST_KEY", rpm=0)
    calls = []

    def _boom(*a, **k):
        calls.append(1)
        raise _http_error(429, body)

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", _boom)
    with pytest.raises(llm_client.QuotaExhausted, match="daily request quota"):
        c._post([{"role": "user", "content": "x"}])
    assert len(calls) == 1     # not self.max_retries


def test_per_minute_429_is_retried(monkeypatch):
    monkeypatch.setenv("COOK_AD_TEST_KEY", "k")
    c = llm_client.ChatClient(api_key_env="COOK_AD_TEST_KEY", rpm=0, max_retries=3)
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
    calls = []

    def _boom(*a, **k):
        calls.append(1)
        raise _http_error(429, '{"error":{"message":"rate limit, requests per minute"}}')

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", _boom)
    with pytest.raises(llm_client.LLMError) as exc:
        c._post([{"role": "user", "content": "x"}])
    assert not isinstance(exc.value, llm_client.QuotaExhausted)
    assert len(calls) == 3
    assert "cached" in str(exc.value)   # tells the user rerunning resumes


def test_rate_limiter_paces_under_the_stated_limit():
    """Spacing at exactly 60/rpm sits dead on the cap and still collects 429s."""
    assert llm_client.RateLimiter(15).min_interval > 60.0 / 15
    assert llm_client.RateLimiter(0).min_interval == 0.0


def test_non_transient_http_error_is_not_retried(monkeypatch):
    monkeypatch.setenv("COOK_AD_TEST_KEY", "k")
    c = llm_client.ChatClient(api_key_env="COOK_AD_TEST_KEY", rpm=0)
    calls = []

    def _boom(*a, **k):
        calls.append(1)
        raise _http_error(404, '{"error":{"message":"model not found"}}')

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", _boom)
    with pytest.raises(llm_client.LLMError, match="model not found"):
        c._post([{"role": "user", "content": "x"}])
    assert len(calls) == 1


# ---- local servers ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "http://localhost:11434/v1", "http://127.0.0.1:8000/v1", "http://0.0.0.0:8000/v1",
])
def test_local_server_needs_no_api_key(monkeypatch, url):
    """Pointing --base-url at a local inference server should need no other ceremony."""
    monkeypatch.delenv("COOK_AD_TEST_KEY", raising=False)
    c = llm_client.ChatClient(base_url=url, api_key_env="COOK_AD_TEST_KEY")
    assert c._api_key() == "local"


def test_remote_server_still_requires_a_key(monkeypatch):
    monkeypatch.delenv("COOK_AD_TEST_KEY", raising=False)
    c = llm_client.ChatClient(base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                              api_key_env="COOK_AD_TEST_KEY")
    with pytest.raises(llm_client.LLMError):
        c._api_key()


def test_explicit_key_still_wins_for_a_local_server(monkeypatch):
    monkeypatch.setenv("COOK_AD_TEST_KEY", "real-key")
    c = llm_client.ChatClient(base_url="http://localhost:11434/v1", api_key_env="COOK_AD_TEST_KEY")
    assert c._api_key() == "real-key"


# ---- concurrency -----------------------------------------------------------------------------

def test_defaults_are_local_and_unpaced():
    """Local is the default destination: no key, no pacing, concurrency on."""
    c = llm_client.ChatClient()
    assert c._is_local()
    assert c.rpm == 0
    assert c.concurrency == llm_client.LOCAL_CONCURRENCY
    assert c.model.startswith("gemma3")


def test_remote_destination_gets_paced_and_serialised():
    """The opposite defaults, because a hosted free tier is request-capped: concurrency there
    would just convert one 429 into eight."""
    c = llm_client.ChatClient(base_url="https://generativelanguage.googleapis.com/v1beta/openai")
    assert not c._is_local()
    assert c.rpm == llm_client.REMOTE_RPM
    assert c.concurrency == 1


def test_explicit_rpm_and_concurrency_override_the_destination_default():
    c = llm_client.ChatClient(base_url="http://localhost:11434/v1", rpm=30, concurrency=2)
    assert (c.rpm, c.concurrency) == (30, 2)
    c = llm_client.ChatClient(base_url="https://example.com/v1", rpm=0)
    assert c.rpm == 0     # an explicit 0 must not be read as "unset"


def test_complete_many_preserves_input_order_under_concurrency(tmp_path):
    """Verdict k must still be step k's after parallel dispatch."""
    import time as _t
    c = llm_client.ChatClient(base_url="http://localhost:1/v1", cache_dir=tmp_path, concurrency=8)
    order = []

    def fake_post(messages):
        idx = int(messages[0]["content"])
        _t.sleep((10 - idx) * 0.005)   # later items finish FIRST if order is not preserved
        order.append(idx)
        return f"reply-{idx}"

    c._post = fake_post
    out = c.complete_many([[{"role": "user", "content": str(i)}] for i in range(10)])
    assert out == [f"reply-{i}" for i in range(10)]
    assert sorted(order) == list(range(10))
    assert order != list(range(10))    # they really did complete out of order


def test_concurrent_counters_are_not_lost(tmp_path):
    c = llm_client.ChatClient(base_url="http://localhost:1/v1", cache_dir=tmp_path, concurrency=16)
    c._post = lambda messages: "No Anomaly"
    c.complete_many([[{"role": "user", "content": str(i)}] for i in range(200)])
    assert c.n_requests == 200
    assert c.stats()["uncached_requests_needed"] == 200


def test_concurrent_cache_writes_are_all_readable(tmp_path):
    """Write-then-rename: a half-written entry would read back as a corrupt cache miss."""
    import json as _j
    c = llm_client.ChatClient(base_url="http://localhost:1/v1", cache_dir=tmp_path, concurrency=16)
    c._post = lambda messages: "x" * 5000
    msgs = [[{"role": "user", "content": str(i)}] for i in range(60)]
    c.complete_many(msgs)
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 60
    for f in files:
        assert _j.loads(f.read_text())["content"] == "x" * 5000
    assert not list(tmp_path.glob("*.tmp"))     # no temp files left behind


def test_incremental_uses_complete_many():
    steps = _steps(3)
    stub = StubClient(["No Anomaly"] * 3)
    detect.run_trial(stub, "SYS", steps, VOCAB, protocol="incremental")
    assert len(stub.calls) == 3
    assert detect.incremental_messages("SYS", steps)[2][1]["content"].count("\n") == 2


def test_rate_limiter_is_shared_across_threads():
    """N workers must be paced as one stream, or concurrency multiplies the effective rate."""
    import threading as _th
    lim = llm_client.RateLimiter(600)      # 0.11s spacing with the safety margin
    start = time.monotonic()
    threads = [_th.Thread(target=lim.wait) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert time.monotonic() - start >= 4 * lim.min_interval


def test_conversational_ablation_feeds_replies_back():
    """The ablation that isolates protocol from model: it must differ from `incremental` in
    exactly one way -- the assistant turns."""
    steps = _steps(3)
    stub = StubClient(["No Anomaly", "omission Anomaly.", "No Anomaly"])
    verdicts = detect.run_trial(stub, "SYS", steps, VOCAB, protocol="conversational")

    assert len(stub.calls) == 3
    assert [v.is_anomaly for v in verdicts] == [False, True, False]
    assert [len(c) for c in stub.calls] == [2, 4, 6]
    assert stub.calls[2][2] == {"role": "assistant", "content": "No Anomaly"}
    # ...and still never shows a future step
    for i, call in enumerate(stub.calls):
        seen = " ".join(m["content"] for m in call)
        for future in steps[i + 1:]:
            assert textify.render_step(future) not in seen


# ---- format tolerances -----------------------------------------------------------------------

@pytest.mark.parametrize("text,etype,corr", [
    # the literal word "Anomaly" dropped -- the type name alone is unambiguous
    ("Substitution. Correct move would have been take bowl for 9 seconds",
     "substitution", ("take", "bowl", 9)),
    ("repetition", "repetition", None),
    # verb_noun with an underscore: the with-recipes preprompt renders recipe steps that way, so
    # models copy it. Rejecting it cost that arm a 25% parse-failure rate.
    ("Omission Anomaly. Correct move would have been stall_kitchen for 4 seconds",
     "omission", ("stall", "kitchen", 4)),
    ("Abandonment. Correct move would have been pour_milk for 12 seconds",
     "abandonment", ("pour", "milk", 12)),
])
def test_format_drift_is_tolerated_not_penalised(text, etype, corr):
    v = detect.parse_response(text, 0, VOCAB)
    assert v.parse_ok is True
    assert v.is_anomaly is True
    assert v.error_type == etype
    assert v.correction == corr


def test_tolerance_does_not_admit_arbitrary_first_words():
    """Making 'Anomaly' optional must not let any leading noun through -- _canon_type still gates
    on the five real types."""
    for text in ("hallucination Anomaly.", "Banana.", "the person seems fine"):
        v = detect.parse_response(text, 0, VOCAB)
        assert v.error_type is None
        assert v.parse_ok is False


def test_no_anomaly_still_wins_over_the_relaxed_type_pattern():
    for text in ("No Anomaly", "no anomaly.", "  NO ANOMALY  "):
        v = detect.parse_response(text, 0, VOCAB)
        assert (v.is_anomaly, v.parse_ok) == (False, True)
