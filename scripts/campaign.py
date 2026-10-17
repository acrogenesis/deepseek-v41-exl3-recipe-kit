#!/usr/bin/env python3
"""Authorized GPU campaign: wait for verified files, smoke, then compare caches."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.preflight import check


def compare(baseline, candidate):
    # A cache-size change must not change generated tokens or exact-check scores.
    a, b = baseline["runs"], candidate["runs"]
    identical = len(a) == len(b) and all((x["id"], x["phase"], x["token_ids"], x.get("passed")) ==
        (y["id"], y["phase"], y["token_ids"], y.get("passed")) for x, y in zip(a, b))
    ratio = candidate["fresh_median_tok_s"] / baseline["fresh_median_tok_s"]
    return {"same_generated_tokens_and_checks": identical, "fresh_speed_ratio": ratio,
        "baseline_fresh_tok_s": baseline["fresh_median_tok_s"], "candidate_fresh_tok_s": candidate["fresh_median_tok_s"],
        "baseline_repeat_tok_s": baseline["repeat_median_tok_s"], "candidate_repeat_tok_s": candidate["repeat_median_tok_s"],
        "recommended_cache_gib": candidate["profile"]["gpu_expert_cache_bytes"] // 2**30 if identical and ratio >= 1.05 else 16,
        "limitation": "One paired sweep, 3 short fresh prompts each; Engram OS page cache may warm between processes. No automatic default change."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    if not args.allow_gpu:
        raise SystemExit("Explicit GPU authorization required")
    ident = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-cache-campaign"
    directory = ROOT / "results" / ident
    directory.mkdir()
    state_path = ROOT / "work/gpu-campaign.json"
    state = {"status": "waiting_for_download", "directory": str(directory), "authorized": True, "steps": []}
    process = None

    def save():
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(state_path)
        (directory / "campaign.json").write_text(json.dumps(state, indent=2) + "\n")

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Stopped by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def hold_check():
        if (ROOT / "work/GPU_HOLD").exists():
            raise RuntimeError("GPU_HOLD requested; stopping campaign")

    def free_gpu():
        while True:
            hold_check()
            free = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).splitlines()
            if len(free) != 1:
                raise RuntimeError("Expected one GPU")
            if int(free[0]) >= 30000:
                return
            state["status"] = "waiting_for_gpu_headroom"
            save()
            time.sleep(20)

    def run(mode, gib):
        nonlocal process
        free_gpu()
        name = f"{mode}-{gib}g"
        state.update(status="running", current_step=name)
        save()
        print(f"Starting {name}", flush=True)
        target = directory / (name + ".json")
        command = ["bash", "scripts/run.sh", "--gpu-is-free", mode, "--gpu-cache-gib", str(gib),
            "--output", f"/recipe/results/{ident}/{name}.json"]
        with (directory / (name + ".log")).open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 5400
            while process.poll() is None:
                hold_check()
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{name} exceeded 90 minutes")
                time.sleep(5)
            code = process.returncode
            process = None
        data = json.loads(target.read_text()) if target.exists() else {"status": "failed", "error": f"exit {code}; see log"}
        state["steps"].append({"name": name, "exit_code": code, "result": str(target), "status": data["status"]})
        save()
        return code, data

    try:
        save()
        while not check()["ready_for_gpu_validation"]:
            hold_check()
            receipt = ROOT / "work/download-state.json"
            if receipt.exists() and json.loads(receipt.read_text()).get("status") == "failed":
                raise RuntimeError("Download failed; refusing GPU campaign")
            failed = subprocess.check_output(["systemctl", "--user", "show", "deepseek41-exl3-download.service",
                "deepseek41-exl3-finalize.service", "-p", "ActiveState"], text=True)
            if "ActiveState=failed" in failed:
                raise RuntimeError("Staging service failed; refusing GPU campaign")
            time.sleep(20)
        code, data = run("smoke", 16)
        if code:
            raise RuntimeError("16 GiB smoke failed: " + data.get("error", "unknown"))
        candidate = 22
        code, data = run("smoke", candidate)
        if code:
            error = data.get("error", "")
            if "OutOfMemoryError" not in error and "out of memory" not in error.lower():
                raise RuntimeError("22 GiB smoke failed: " + error)
            candidate = 20
            code, data = run("smoke", candidate)
            if code:
                raise RuntimeError("20 GiB smoke failed: " + data.get("error", "unknown"))
        code, baseline = run("bench", 16)
        if code:
            raise RuntimeError("Baseline benchmark failed")
        code, larger = run("bench", candidate)
        if code:
            raise RuntimeError("Larger-cache benchmark failed")
        state["comparison"] = compare(baseline, larger)
        state.update(status="complete", current_step=None)
        print(json.dumps(state["comparison"], indent=2), flush=True)
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        raise
    finally:
        if process is not None and process.poll() is None:
            subprocess.run(["podman", "stop", "--time", "10", "deepseek41-exl3-5090"], check=False)
            process.wait(timeout=30)
        save()


if __name__ == "__main__":
    main()
