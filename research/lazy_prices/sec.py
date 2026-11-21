"""Polite, rate-limited access to SEC EDGAR.

SEC requires a User-Agent that names the requester and gives a contact email
(https://www.sec.gov/os/accessing-edgar-data). Provide it either as the
environment variable SEC_USER_AGENT or as the single line of ~/.sec_user_agent:

    Your Name your.email@example.edu

Never commit it.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import requests

MAX_RPS = 8.0  # SEC's limit is 10 requests/second


def user_agent() -> str:
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    f = Path.home() / ".sec_user_agent"
    if not ua and f.exists():
        ua = f.read_text().strip()
    if "@" not in ua:
        raise RuntimeError("SEC requires a contact email: set SEC_USER_AGENT or write "
                           "'Your Name email@domain' to ~/.sec_user_agent")
    return ua


class _RateLimiter:
    def __init__(self, rps: float):
        self.gap = 1.0 / rps
        self.lock = threading.Lock()
        self.next = time.monotonic()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.gap
        time.sleep(max(0.0, t - now))


_limiter = _RateLimiter(MAX_RPS)
_local = threading.local()


def _session() -> requests.Session:
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": user_agent(), "Accept-Encoding": "gzip, deflate"})
        _local.s = s
    return s


def get(url: str, retries: int = 5, timeout: float = 90.0) -> requests.Response:
    for attempt in range(retries):
        _limiter.wait()
        try:
            r = _session().get(url, timeout=timeout)
        except requests.RequestException:
            time.sleep(2**attempt)
            continue
        if r.status_code == 200:
            return r
        if r.status_code in (403, 429, 500, 502, 503):
            time.sleep(2 ** (attempt + 1))
            continue
        r.raise_for_status()
    raise RuntimeError(f"failed after {retries} tries: {url}")
