"""Groq key pool. Labels are key_1, key_2, ... The key values are never logged."""

from __future__ import annotations

import os


class KeyPool:
    def __init__(self, keys: list[str]) -> None:
        self.keys = [key for key in keys if key]
        self.index = 0
        if self.keys:
            os.environ["GROQ_API_KEY"] = self.keys[0]

    def label(self) -> str:
        if not self.keys:
            return "key_1"
        return f"key_{self.index + 1}"

    def current(self) -> str:
        if not self.keys:
            raise RuntimeError("GROQ_API_KEY is not set. Add GROQ_API_KEYS or GROQ_API_KEY to .env.")
        return self.keys[self.index]

    def rotate(self) -> bool:
        """Move to the next key after a daily cap. Returns False when none remain."""
        if self.index + 1 >= len(self.keys):
            return False
        self.index += 1
        os.environ["GROQ_API_KEY"] = self.keys[self.index]
        return True

    @classmethod
    def from_env(cls) -> "KeyPool":
        raw = os.environ.get("GROQ_API_KEYS") or ""
        keys = [part.strip() for part in raw.split(",") if part.strip()]
        if not keys:
            single = (os.environ.get("GROQ_API_KEY") or "").strip()
            keys = [single] if single else []
        return cls(keys)


_POOL: KeyPool | None = None


def key_pool(*, reset: bool = False) -> KeyPool:
    global _POOL
    if _POOL is None or reset:
        _POOL = KeyPool.from_env()
    return _POOL


def fit_daily_cap(output_dir, estimate: int, cap: int) -> tuple[bool, object]:
    """Pick a key whose own usage file can take `estimate`. Labels only; the key is not returned."""
    from pathlib import Path

    from src.vision.usage import tokens_used, usage_path

    pool = key_pool()
    root = Path(output_dir)
    last = usage_path(root)
    slots = max(len(pool.keys), 1)
    for index in range(slots):
        api_key = pool.current() if pool.keys else None
        path = usage_path(root, api_key=api_key)
        last = path
        if tokens_used(path) + int(estimate) <= int(cap):
            return True, path
        if index == slots - 1 or not pool.rotate():
            return False, path
    return False, last


def scrub(text: str) -> str:
    """Remove any loaded key from a string before it is logged or stored."""
    cleaned = text or ""
    pool = key_pool()
    for key in pool.keys:
        if key and key in cleaned:
            cleaned = cleaned.replace(key, "[redacted]")
    return cleaned
