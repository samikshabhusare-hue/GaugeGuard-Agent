#!/usr/bin/env python3
"""GaugeGuard Evaluation CLI Entrypoint.

Usage:
    python evaluate.py --labels data/samples/labels.csv --images-dir data/samples/
    python evaluate.py --tolerance 2.5 --output-dir evaluation_results/
    python evaluate.py --tolerance-fraction 0.05
"""

import sys
from gaugeguard.evaluation import main

if __name__ == "__main__":
    sys.exit(main())
