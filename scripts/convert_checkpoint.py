#!/usr/bin/env python3
"""Convert one experiment JSON artifact into the public SoT `.pt` format."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from sot.checkpoint import checkpoint_summary, save_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--json-output",
        type=Path,
        help="optionally copy the exact human-readable source artifact after a privacy check",
    )
    parser.add_argument("--release-id", required=True)
    args = parser.parse_args()
    save_checkpoint(args.artifact, args.output, release_id=args.release_id)
    if args.json_output is not None:
        text = args.artifact.read_text(encoding="utf-8")
        private_path = re.compile("(?:/" + "scratch/|/" + "home/|/" + "Users/)")
        if private_path.search(text):
            raise ValueError("source artifact contains a machine-specific absolute path")
        # Parse before copying so malformed JSON cannot enter the release bundle.
        json.loads(text)
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text, encoding="utf-8")
    print(json.dumps(checkpoint_summary(args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
