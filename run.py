from __future__ import annotations

import sys

from longspeech_eval.cli import main


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].startswith("-"):
        sys.argv.insert(1, "run")
    main()
