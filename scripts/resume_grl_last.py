#!/usr/bin/env python3
"""Resume the held GRL jobs only after groups 1 and 2 finish completely."""
from __future__ import annotations

import fcntl
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
G1 = ROOT / "results/campaigns/c005_official_group1/jobs.jsonl"
G2 = ROOT / "results/campaigns/c005_official_group2/jobs.jsonl"
G3 = ROOT / "results/campaigns/c005_official_group3/jobs.jsonl"
SPEC3 = ROOT / "scripts/campaigns/c005_official_group3.json"
LOG3 = ROOT / "results/campaigns/c005_official_group3/campaign_run.log"
CONT_ID = "train:ours_dim48_anchor:13:continue20k"


def read_jobs(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def complete(path: Path) -> bool:
    if not path.exists():
        return False
    jobs = read_jobs(path)
    return bool(jobs) and all(job["status"] == "done" for job in jobs)


def continuation_done() -> bool:
    if not G3.exists():
        return False
    return any(job["id"] == CONT_ID and job["status"] == "done" for job in read_jobs(G3))


while not (complete(G1) and complete(G2) and continuation_done()):
    time.sleep(30)

with G3.open("r+", encoding="utf-8") as fh:
    fcntl.flock(fh, fcntl.LOCK_EX)
    jobs = read_jobs(G3)
    changed = False
    for job in jobs:
        if job["status"] == "held" and job.get("model") == "grl":
            job["status"] = "pending"
            changed = True
    if changed:
        fh.seek(0)
        fh.truncate()
        fh.write("".join(json.dumps(job, ensure_ascii=False) + "\n" for job in jobs))
        fh.flush()
    fcntl.flock(fh, fcntl.LOCK_UN)

if changed:
    with LOG3.open("ab") as log:
        subprocess.Popen(
            ["bash", str(ROOT / "scripts/run_controlled_group.sh"), str(SPEC3)],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
