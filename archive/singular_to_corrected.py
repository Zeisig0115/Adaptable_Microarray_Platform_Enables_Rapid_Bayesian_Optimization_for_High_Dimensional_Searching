"""Thin entry point for the canonical mixed-data loader.

All schema detection, objective extraction, and output writing now live in
``data_loader_objectives.py`` so the two command names cannot drift apart.
"""

import sys

from data_loader_objectives import main


if __name__ == "__main__":
    has_inputs = any(
        arg == "--inputs" or arg.startswith("--inputs=") for arg in sys.argv[1:]
    )
    if not {"-h", "--help"}.intersection(sys.argv[1:]) and not has_inputs:
        raise SystemExit(
            "singular_to_corrected.py now uses the unified loader; pass one or more "
            "explicit paths after --inputs."
        )
    main()
