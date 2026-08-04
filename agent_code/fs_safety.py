from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# Allowed text file extension / suffix
TEXT_SUFFIXES = {
    ".py", ".pyi", ".md", ".rst", ".txt", ".toml", ".yaml", ".yml", ".json",
    ".cfg", ".ini", ".env", ".sh", ".bash", ".zsh", ".js", ".ts", ".tsx",
    ".jsx", ".html", ".css", ".sql", ".lock", ".gitignore",
}

# Max size of file
MAX_READ_BYTES = 256 * 1024

# Max length of tool observations
DEFAULT_MAX_CHARS = 8000

DEFAULT_SKIP_DIRS = frozenset({
    ".git", ".venv", "venv", "node_modules", "dist", "build",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
})


@dataclass
class SkipPolicy:
    skip_dirs: frozenset[str] = DEFAULT_SKIP_DIRS

    @classmethod
    def default(cls) -> "SkipPolicy":
        return cls()


@dataclass
class ReadFileState:
    entries: dict[Path, tuple[int, int]] = field(default_factory=dict)

    def record(self, path: Path, content: str) -> None:
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            return
        self.entries[path] = (mtime_ns, len(content))


def resolve_in_cwd(cwd: Path, user_path: str) -> Path:
    candidate = (cwd / user_path).resolve()
    cwd_resolved = cwd.resolve()
    try:
        candidate.relative_to(cwd_resolved)
    except ValueError as exc:
        raise ValueError(f"path escapes cwd: {user_path}") from exc
    return candidate

def ensure_text_file(path: Path) -> None:
    if path.suffix.lower() in TEXT_SUFFIXES:
        return
    with path.open("rb") as f:
        if b"\x00" in f.read(1024):
            raise ValueError(f"binary file: {path.name}")

def ensure_within_size(path: Path, max_bytes: int = MAX_READ_BYTES) -> None:
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(
            f"file too large: {size} bytes > {max_bytes};"
            f"read a smaller file or use grep instead"
        )

def should_skip(rel_path: Path, policy: SkipPolicy) -> bool:
    return any(part in policy.skip_dirs for part in rel_path.parts)

def truncate_output(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n[truncated {len(text) - max_chars} chars]"