"""Legacy entry point - use `tennis-analyze` CLI or `python cli.py` instead."""

from cli import main

if __name__ == "__main__":
    main()
else:
    # Backwards compatibility: running as `python main.py` without __main__ guard
    main()
