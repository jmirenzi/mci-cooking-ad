"""Minimal OpenAI-compatible chat client for the LLM baseline.

Deliberately stdlib-only (urllib.request): this is one JSON POST, and pyproject.toml's dependency
list is kept lean on purpose. Nothing here is provider-specific beyond the defaults -- Google's
OpenAI-compatible endpoint, OpenRouter, and a local server all accept the same /chat/completions
body, so switching providers is a --base-url flag rather than a code change.

The default is the Gemini API rather than OpenRouter, and the reason is the request budget. This
evaluation is request-bound, not token-bound: at ~7 steps per trial the incremental protocol costs
~7 requests per trial, so a 6-condition sweep (healthy + 5 error types) over 20 trials is ~880
requests per prompt variant.

  * Gemini free tier, Flash-Lite: 15 requests/minute and 1000 requests/DAY were verified for
    gemini-3.5-flash-lite; DEFAULT_RPM is set to that floor. Newer Flash-Lite revisions have been
    reported at up to 30 RPM, so check your own limit at
    https://aistudio.google.com/rate-limit and raise --rpm if yours is higher -- the token bucket
    only ever slows requests down, so the conservative default is safe but not optimal. A
    20-trial single-variant incremental sweep fits inside one 1000-request day.
  * OpenRouter :free variants: 20 requests/minute but only 50 requests/day, rising to 1000/day
    only after $10 of lifetime credit purchases. That cap is per-account and platform-wide -- it
    does NOT depend on which free model you pick, so choosing a smaller model buys nothing.

Switch back with:
    --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY \
    --model <something>:free --rpm 20

Either way the on-disk cache below makes reruns and resumed sweeps free, which is what lets a
sweep survive a daily cap being hit part-way through.
"""
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

# Google's OpenAI-compatibility layer. Same /chat/completions body as everyone else; the trailing
# path segment is "openai", and ChatClient appends "/chat/completions".
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEFAULT_API_KEY_ENV = "GEMINI_API_KEY"
# Small on purpose: this is a proof-of-concept baseline, and Flash-Lite carries the friendliest
# free-tier request budget (15 RPM / 1000 RPD), which is the binding constraint here.
DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_RPM = 15            # verified Flash-Lite free-tier floor; raise if your tier allows
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 5

# Untracked file holding the API key. Listed in .gitignore; .env.example shows the shape.
DEFAULT_ENV_FILE = ".env"


class BudgetExceeded(RuntimeError):
    """Raised when a sweep would exceed --max-requests. Raised rather than returning a partial
    result on purpose: a half-finished sweep that silently reports as complete is worse than a
    crash, and the cache means restarting after raising the budget costs nothing."""


class LLMError(RuntimeError):
    pass


def load_env_file(path=DEFAULT_ENV_FILE, override=False):
    """Populate os.environ from a KEY=VALUE file, so the API key lives in an untracked file
    instead of being exported by hand every session.

    Hand-rolled rather than adding python-dotenv: this is a dozen lines and the dependency list is
    deliberately lean. A missing file is fine (returns 0) -- exporting the variable directly is
    equally valid, and CI has no .env. Existing environment variables win by default, so an
    explicit `GEMINI_API_KEY=... python run_llm_eval.py` is never silently overridden by a stale
    file.
    """
    p = Path(path)
    if not p.exists():
        return 0
    n = 0
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and (override or key not in os.environ):
            os.environ[key] = value
            n += 1
    return n


class RateLimiter:
    """Token bucket at `rpm` requests/minute. Cache hits do not consume budget (they never leave
    the process), so this only paces real calls."""

    def __init__(self, rpm=DEFAULT_RPM):
        self.min_interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._last = 0.0

    def wait(self):
        if self.min_interval <= 0:
            return
        delta = time.monotonic() - self._last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last = time.monotonic()


class ChatClient:
    """Cached, rate-limited, budget-guarded chat completions.

    `dry_run=True` performs no network I/O at all: it counts what a sweep WOULD cost and returns a
    fixed placeholder reply. That is the first thing to run against an unfamiliar request budget.
    """

    def __init__(self, model=DEFAULT_MODEL, base_url=DEFAULT_BASE_URL,
                 api_key_env=DEFAULT_API_KEY_ENV, cache_dir=None, rpm=DEFAULT_RPM,
                 max_requests=None, temperature=0.0, timeout=DEFAULT_TIMEOUT,
                 max_retries=DEFAULT_MAX_RETRIES, dry_run=False, extra_headers=None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.dry_run = dry_run
        self.max_requests = max_requests
        self.extra_headers = dict(extra_headers or {})
        self.limiter = RateLimiter(rpm)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.n_requests = 0      # real network calls
        self.n_cache_hits = 0
        self.n_would_request = 0  # what a dry run would have cost

    # ---- key handling -------------------------------------------------------------------

    def _api_key(self):
        key = os.environ.get(self.api_key_env)
        if not key:
            raise LLMError(
                f"no API key in ${self.api_key_env}. Put it in an untracked {DEFAULT_ENV_FILE} "
                f"file as '{self.api_key_env}=...' (copy .env.example), or export it, or pass "
                f"--api-key-env with the name of the variable holding it. --dry-run costs a sweep "
                f"without needing a key at all."
            )
        return key

    # ---- cache --------------------------------------------------------------------------

    def _cache_key(self, messages):
        payload = json.dumps(
            {"base_url": self.base_url, "model": self.model, "temperature": self.temperature,
             "messages": messages},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_path(self, key):
        return self.cache_dir / f"{key}.json" if self.cache_dir else None

    def _cache_get(self, key):
        path = self._cache_path(key)
        if path is None or not path.exists():
            return None
        try:
            return json.loads(path.read_text())["content"]
        except (json.JSONDecodeError, KeyError, OSError):
            return None  # a corrupt entry is a cache miss, not a crash

    def _cache_put(self, key, content, messages):
        path = self._cache_path(key)
        if path is None:
            return
        path.write_text(json.dumps(
            {"content": content, "model": self.model, "base_url": self.base_url,
             "temperature": self.temperature, "messages": messages,
             "written_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
        ))

    # ---- the call -----------------------------------------------------------------------

    def _post(self, messages):
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }).encode()
        headers = {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body,
                                     headers=headers, method="POST")

        delay = 1.0
        last_err = None
        for attempt in range(self.max_retries):
            self.limiter.wait()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode())
                return payload["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                # 429 and 5xx are transient; everything else (401, 400, 404 model-not-found) is a
                # configuration error that retrying cannot fix, so fail immediately with the body.
                if e.code != 429 and e.code < 500:
                    raise LLMError(f"HTTP {e.code} from {self.base_url}: {e.read().decode()[:500]}") from e
                retry_after = e.headers.get("Retry-After") if e.headers else None
                sleep_for = float(retry_after) if retry_after and retry_after.isdigit() else delay
                last_err = e
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError) as e:
                sleep_for = delay
                last_err = e
            if attempt < self.max_retries - 1:
                time.sleep(sleep_for)
                delay = min(delay * 2, 60.0)
        raise LLMError(f"giving up after {self.max_retries} attempts: {last_err!r}")

    def complete(self, messages):
        """messages: OpenAI-format [{'role','content'}, ...]. Returns the assistant text."""
        key = self._cache_key(messages)
        cached = self._cache_get(key)
        if cached is not None:
            self.n_cache_hits += 1
            return cached

        self.n_would_request += 1
        if self.max_requests is not None and self.n_would_request > self.max_requests:
            raise BudgetExceeded(
                f"this sweep needs more than --max-requests={self.max_requests} uncached "
                f"requests. Raise the budget and rerun -- cached responses are reused, so the "
                f"work already done is not repeated."
            )
        if self.dry_run:
            return "No Anomaly"

        content = self._post(messages)
        self.n_requests += 1
        self._cache_put(key, content, messages)
        return content

    def stats(self):
        return {
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "requests_sent": self.n_requests,
            "cache_hits": self.n_cache_hits,
            "uncached_requests_needed": self.n_would_request,
            "dry_run": self.dry_run,
        }
