#!/usr/bin/env python3
"""Hold one pending campaign job without touching running jobs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("job_id")
    args = parser.parse_args()
    jobs = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    found = False
    for job in jobs:
        if job["id"] == args.job_id:
            if job["status"] == "pending":
                job["status"] = "held"
            found = True
    if not found:
        raise SystemExit(f"job not found: {args.job_id}")
    args.manifest.write_text("".join(json.dumps(job, ensure_ascii=False) + "\n" for job in jobs))
    print(f"held: {args.job_id}")


if __name__ == "__main__":
    main()
