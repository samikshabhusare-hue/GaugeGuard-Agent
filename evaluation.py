#!/usr/bin/env python3
"""GaugeGuard Evaluation Runner CLI Entrypoint.

Usage:
    python evaluation.py --run --out evaluation_results/local/
    python evaluation.py --labels data/samples/labels.csv --images-dir data/samples/
"""

import sys
from gaugeguard.evaluation import (
    load_image_bgr,
    run_pipeline,
    StubPipeline,
    capture_environment,
    compute_baseline_metrics,
    run_evaluation,
    main,
)

if __name__ == "__main__":
    sys.exit(main())
