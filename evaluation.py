#!/usr/bin/env python3
"""GaugeGuard Evaluation Runner CLI Entrypoint.

Usage:
    python evaluation.py --run --out evaluation_results/local/
    python evaluation.py --report evaluation_results/local/
    python evaluation.py --compare evaluation_results/local/ evaluation_results/ec2/
"""

import sys
from gaugeguard.evaluation import (
    load_image_bgr,
    run_pipeline,
    StubPipeline,
    capture_environment,
    compute_baseline_metrics,
    run_evaluation,
    make_report,
    compare_runs,
    main,
)

if __name__ == "__main__":
    sys.exit(main())
