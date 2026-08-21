from __future__ import annotations

import argparse
from pathlib import Path

import certify_theta_symmetric_quandle as sq
import certify_theta_symmetric_quandle_v2 as sq2
import validate_theta_symmetric_quandle_invariance as validation

# Patch the base module used by the validation harness.
sq.diagram_constraints = sq2.corrected_diagram_constraints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plantri", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validation.run(args.plantri, args.output)


if __name__ == "__main__":
    main()
