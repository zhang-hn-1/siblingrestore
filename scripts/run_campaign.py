#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
正式实验 Campaign 编排器（对应 LARGE_SCALE_EXPERIMENT_DESIGN.md 第 6 节）。

子命令：
  prepare <campaign.json>   从模板 configs 生成 campaign 配置 + jobs manifest（幂等）
  run <campaign.json>       在指定 GPU 上循环执行未完成 job（幂等、可断点）
  status <campaign.json>    打印 manifest 汇总

campaign.json 示例（scripts/campaigns/c001_sfr_v1.json）：
{
  "name": "c001_sfr_v1",
  "data_root": "data/plamd_sfr_v1",
  "template_config_dir": "configs/baselines_500",
  "verifier_template": "verifier_evaluator.json",
  "verifier_source_count": 895,
  "models": ["restormer", "promptir", "prenet", "grl",
             "ours_dim48_anchor", "ours_dim48", "ours_two_stage_v2",
             "ours_dim32_msc", "ours_dim48_two_stage_anchor", "ours_dim64", "ours_dim64_anchor"],
  "seeds": [13],
  "splits": ["val", "test"],
  "gpus": [0, 1, 2, 3],
  "lpips": true
}

依赖规则：verifier → train(model,seed) → eval(model,seed,split)。
幂等：done 且指纹一致自动跳过；eval 的 verifier sha 与报告内指纹不一致会自动重跑。
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha12(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------
def cmd_prepare(spec_path: Path) -> None:
    spec = load_json(spec_path)
    name = spec["name"]
    data_root = spec["data_root"]
    config_dir = Path("configs") / "campaigns" / name
    run_root = Path("runs") / "campaigns" / name
    result_root = Path("results") / "campaigns" / name
    log_dir = result_root / "logs"
    manifest_path = result_root / "jobs.jsonl"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    template_dir = Path(spec["template_config_dir"])
    jobs = []
    verifier_cfg_out = config_dir / "verifier_evaluator.json"
    if not verifier_cfg_out.exists():
        cfg = load_json(template_dir / spec["verifier_template"])
        cfg["data_root"] = data_root
        cfg["output_dir"] = str(run_root / "verifier_evaluator")
        cfg["model"]["source_count"] = int(spec["verifier_source_count"])
        verifier_cfg_out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    verifier_best = run_root / "verifier_evaluator" / "best.pt"
    shared_verifier = Path(spec["shared_verifier"]) if spec.get("shared_verifier") else None
    shared_verifier_ready = bool(shared_verifier and shared_verifier.exists())
    if shared_verifier_ready:
        verifier_best = shared_verifier
    jobs.append({"id": "verifier", "type": "verifier",
                 "cmd": [PYTHON, str(PROJECT_ROOT / "train_verifier.py"), "--config", str(verifier_cfg_out)],
                 "depends": [],
                 "initial_status": "done" if shared_verifier_ready else "pending"})

    model_jobs = []
    for model in spec["models"]:
        src_cfg = template_dir / model
        if not src_cfg.exists():
            # Legacy/supplementary baselines live under configs/baselines,
            # while the formal 500/full-data configs live under the campaign
            # template directory.
            legacy_cfg = Path("configs") / "baselines" / model
            if legacy_cfg.exists():
                src_cfg = legacy_cfg
        cfgs = sorted(src_cfg.glob("*.json")) if src_cfg.is_dir() else [src_cfg]
        if not cfgs:
            print(f"[warn] 模板缺失：{src_cfg}，跳过 {model}")
            continue
        cfg_out = config_dir / f"{model}.json"
        if not cfg_out.exists():
            cfg = load_json(cfgs[0])
            cfg["data_root"] = data_root
            for key, value in spec.get("overrides", {}).items():  # 统一训练预算（论文公平性）
                if key in cfg or key == "max_steps" or key == "crop_size":
                    cfg[key] = value
            cfg_out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for seed in spec["seeds"]:
            model_jobs.append({"model": model, "seed": seed})
            jobs.append({"id": f"train:{model}:{seed}", "type": "train", "model": model, "seed": seed,
                         "cmd": [PYTHON, str(PROJECT_ROOT / "train.py"), "--config", str(cfg_out),
                                 "--output-dir", str(run_root / model / str(seed)),
                                 "--seed", str(seed)],
                         "depends": []})
    for model, seed in [(j["model"], j["seed"]) for j in model_jobs]:
        for split in spec["splits"]:
            jobs.append({"id": f"eval:{model}:{seed}:{split}", "type": "eval",
                         "model": model, "seed": seed, "split": split,
                         "cmd": [PYTHON, str(PROJECT_ROOT / "scripts" / "evaluate_frozen_verifier_v2.py"),
                                 "--checkpoint", str(run_root / model / str(seed) / "best.pt"),
                                 "--verifier", str(verifier_best),
                                 "--data-root", data_root,
                                 "--output", str(result_root / f"{model}.{seed}.{split}.json"),
                                 "--split", split, "--device", "cuda",
                                 "--method", model, "--seed", str(seed)] +
                                (["--lpips"] if spec.get("lpips") else []),
                         "depends": ["verifier", f"train:{model}:{seed}"]})

    existing = {j["id"]: j for j in (json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines())} if manifest_path.exists() else {}
    with manifest_path.open("a", encoding="utf-8") as fh:
        for job in jobs:
            if job["id"] in existing:
                continue
            fh.write(json.dumps({"id": job["id"], "type": job["type"],
                                 "model": job.get("model"), "seed": job.get("seed"),
                                 "split": job.get("split"), "cmd": job["cmd"],
                                 "depends": job["depends"],
                                 "status": job.get("initial_status", "pending"), "gpu": None, "exit_code": None,
                                 "started_utc": None, "finished_utc": None}) + "\n")
    print(f"prepare 完成：{len(jobs)} jobs → {manifest_path}")


# ---------------------------------------------------------------------------
# manifest 读写（文件锁保护）
# ---------------------------------------------------------------------------
def read_manifest(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_manifest(path: Path, jobs: list[dict]) -> None:
    path.write_text("".join(json.dumps(j, ensure_ascii=False) + "\n" for j in jobs), encoding="utf-8")


class FileLock:
    def __init__(self, path: Path):
        self.fh = open(path, "a+")

    def __enter__(self):
        fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
        self.fh.close()


def is_done(job: dict, spec: dict) -> bool:
    if job["status"] == "done":
        return True
    # 已产出但状态未更新（崩溃后）→ 重跑；eval 额外校验指纹
    if job["type"] == "verifier":
        return Path(spec["verifier_best"]).exists() if "verifier_best" in spec else False
    return False


def pick_job(jobs: list[dict], done_ids: set) -> dict | None:
    for job in jobs:
        if job["status"] in ("done", "running", "failed", "held"):
            # failed job must be repaired/reset explicitly; automatically
            # retrying it can create an unbounded loop (e.g. CUDA unavailable
            # or an out-of-memory baseline).
            continue
        if job["depends"] and any(d not in done_ids for d in job["depends"]):
            continue
        return job
    return None


def worker(gpu: int, spec: dict, manifest_path: Path, log_dir: Path) -> None:
    lock_path = manifest_path.with_name(f"gpu{gpu}.lock")
    while True:
        with FileLock(manifest_path):
            jobs = read_manifest(manifest_path)
            done_ids = {j["id"] for j in jobs if j["status"] == "done"}
            job = pick_job(jobs, done_ids)
            if job is None:
                return
            job["status"] = "running"
            job["gpu"] = gpu
            job["started_utc"] = utcnow()
            write_manifest(manifest_path, jobs)
        job_id = job["id"]
        log_path = log_dir / f"{job_id}.log"
        env_cmd = ["/usr/bin/env", "CUDA_VISIBLE_DEVICES=%d" % gpu, "PYTHONUNBUFFERED=1"] + job["cmd"]
        print(f"[gpu{gpu}] start {job_id}", flush=True)
        with open(log_path, "ab") as fh:
            proc = subprocess.run(env_cmd, cwd=str(PROJECT_ROOT), stdout=fh, stderr=subprocess.STDOUT)
        with FileLock(manifest_path):
            jobs = read_manifest(manifest_path)
            for j in jobs:
                if j["id"] == job_id:
                    j["status"] = "done" if proc.returncode == 0 else "failed"
                    j["exit_code"] = proc.returncode
                    j["finished_utc"] = utcnow()
            write_manifest(manifest_path, jobs)
        print(f"[gpu{gpu}] {'done' if proc.returncode == 0 else 'FAILED(%d)' % proc.returncode} {job_id}", flush=True)
        if proc.returncode != 0:
            # Strict group control: do not skip a failed model and start the
            # next one without an explicit repair/retry decision.
            return


def cmd_run(spec_path: Path) -> None:
    spec = load_json(spec_path)
    name = spec["name"]
    result_root = Path("results") / "campaigns" / name
    manifest_path = result_root / "jobs.jsonl"
    log_dir = result_root / "logs"
    assert manifest_path.exists(), "先运行 prepare"
    # 上次崩溃残留的 running 状态重置为 pending（本进程内由文件锁保证不重复领取）
    with FileLock(manifest_path):
        jobs = read_manifest(manifest_path)
        for j in jobs:
            if j["status"] == "running":
                j["status"] = "pending"
                j["gpu"] = None
        write_manifest(manifest_path, jobs)
    import multiprocessing
    workers = [multiprocessing.Process(target=worker, args=(gpu, spec, manifest_path, log_dir), daemon=True)
               for gpu in spec["gpus"]]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    jobs = read_manifest(manifest_path)
    from collections import Counter
    print(Counter(j["status"] for j in jobs))


def cmd_status(spec_path: Path) -> None:
    spec = load_json(spec_path)
    name = spec["name"]
    manifest_path = Path("results") / "campaigns" / name / "jobs.jsonl"
    if not manifest_path.exists():
        print("manifest 不存在，先 prepare")
        return
    jobs = read_manifest(manifest_path)
    print(f"{'id':<34} {'type':<9} {'status':<8} gpu exit")
    for j in jobs:
        print(f"{j['id']:<34} {j['type']:<9} {j['status']:<8} {j['gpu']} {j['exit_code']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, func in (("prepare", cmd_prepare), ("run", cmd_run), ("status", cmd_status)):
        p = sub.add_parser(name)
        p.add_argument("spec", type=Path)
        p.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args.spec)


if __name__ == "__main__":
    main()
