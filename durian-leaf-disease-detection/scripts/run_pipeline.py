#!/usr/bin/env python
"""Script chạy toàn bộ pipeline từ đầu đến cuối.

Usage:
    uv run scripts/run_pipeline.py
    uv run scripts/run_pipeline.py --config config/custom.yaml
    uv run scripts/run_pipeline.py --skip train_model
"""

from durian_detect.cli import main

if __name__ == "__main__":
    import sys

    # Inject "pipeline" command nếu chưa có
    if len(sys.argv) == 1 or sys.argv[1].startswith("-"):
        sys.argv.insert(1, "pipeline")

    main()
