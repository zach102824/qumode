"""python -m system_size_scaling → print the README command block."""

from __future__ import annotations

from .config import ROOT


def main() -> int:
    readme = ROOT / "README.md"
    print(readme.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
