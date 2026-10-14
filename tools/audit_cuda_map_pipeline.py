"""Audit complete frozen map pipelines on the CPU without raw inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.audit_cuda_fixed_neighbors import audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    print(json.dumps(audit(parser.parse_args().archive), indent=2))


if __name__ == "__main__":
    main()
