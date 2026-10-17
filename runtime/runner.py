"""GPU entry point; imports CUDA code only after an explicit execution flag."""
import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["smoke", "bench"])
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    if not args.allow_gpu:
        raise SystemExit("GPU work is on hold. Run scripts/run.sh only after the GPU is free.")
    from scripts.preflight import check
    report = check()
    if not report["ready_for_gpu_validation"]:
        raise SystemExit(json.dumps(report, indent=2))
    for name in ("EXL3_ATS_MMAP", "EXL3_ENGRAM_ATS", "EXL3_MOE_CPU_SPLIT", "EXL3_MOE_CPU_OFFLOAD"):
        os.environ[name] = "0"
    # Keep the first validation on the ordinary EXL3 arithmetic path.
    os.environ["EXL3_INT8_GEMV"] = "0"
    import torch
    from exllamav3 import Config, Tokenizer
    from exllamav3.cache.cache import Cache
    from exllamav3.generator import Generator, Job
    from exllamav3.generator.sampler import GreedySampler
    from .indexed_store import inventory, IndexedExpertStore
    from .cache import ExpertCache
    from .native_adapter import attach_engram, build_model
    from .memory_budget import guard_memory
    profile = json.loads((ROOT / "profiles/5090.json").read_text())
    guard_memory(profile["host_reserve_bytes"])
    if torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("This build/profile targets the SM120 RTX 5090")
    records, inv = inventory(ROOT)
    store = IndexedExpertStore(ROOT / "work/models/exl3", records, profile["ram_expert_cache_bytes"],
        reserve_bytes=profile["host_reserve_bytes"])
    capacity = profile["gpu_expert_cache_bytes"]
    per_layer = capacity // (40 * 4096) * 4096
    quotas = {l: per_layer for l in range(40)}
    quotas[39] += capacity - sum(quotas.values())
    experts = ExpertCache(store, capacity, layer_quotas=quotas, reserve_bytes=profile["host_reserve_bytes"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "results" / f"{stamp}-{args.mode}.json"
    result = {"status": "loading", "profile": profile, "inventory": inv, "runs": [],
        "runtime_lock": json.loads((ROOT / "runtime.lock.json").read_text()),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
        "metric": "Streamed output tokens after first token / elapsed time after first token; fresh KV state each request"}

    def save():
        tmp = output.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        tmp.replace(output)

    model = None
    try:
        save()
        config = Config.from_directory(str(ROOT / "work/models/exl3"))
        config.infer_params.no_reconstruct = True
        attach_engram(config, ROOT / "work/models/engram")
        model = build_model(config, experts)
        kv = Cache(model, max_num_tokens=profile["context_tokens"], max_batch_size=1)
        started = time.perf_counter()
        model.load("cuda:0", progressbar=False)
        result["load_seconds"] = time.perf_counter() - started
        tokenizer = Tokenizer.from_config(config)
        fixture_path = ROOT / "fixtures/paired-gguf.json"
        fixture = json.loads(fixture_path.read_text())
        if fixture["context"] != profile["context_tokens"]:
            raise RuntimeError("Context differs from paired GGUF fixture")
        result["fixture_sha256"] = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
        for case in fixture["checks"] + fixture["speed"]:
            actual = tokenizer.encode(case["rendered_prompt"], add_bos=False, encode_special_tokens=True).flatten().tolist()
            if actual != case["input_ids"]:
                raise RuntimeError(f"Tokenizer differs from paired fixture: {case['id']}")

        def run(case, phase, cap):
            # A new generator creates a fresh page table and resets recurrent state.
            generator = Generator(model=model, cache=kv, tokenizer=tokenizer,
                max_batch_size=1, max_chunk_size=profile["prefill_chunk_tokens"], recurrent_cache_size=0)
            ids = torch.tensor([case["input_ids"]], dtype=torch.long)
            job = Job(input_ids=ids, max_new_tokens=cap, sampler=GreedySampler(),
                stop_conditions=config.eos_token_id_list)
            generator.enqueue(job)
            tokens, chunks, events = [], [], []
            start = time.perf_counter()
            first = last = None
            first_count = 0
            while generator.num_remaining_jobs():
                for event in generator.iterate():
                    if event.get("stage") == "error":
                        raise RuntimeError(str(event))
                    if event.get("stage") != "streaming":
                        continue
                    if event.get("text"):
                        chunks.append(event["text"])
                    emitted = event.get("token_ids")
                    if emitted is None or not emitted.numel():
                        continue
                    torch.cuda.synchronize()
                    now = time.perf_counter() - start
                    if first is None:
                        first, first_count = now, emitted.numel()
                    last = now
                    token_ids = emitted.flatten().tolist()
                    tokens.extend(token_ids)
                    events.append({"seconds": now, "tokens": token_ids})
            duration = time.perf_counter() - start
            rate = (len(tokens) - first_count) / (last - first) if first is not None and last > first else None
            row = {"id": case["id"], "phase": phase, "input_tokens": len(case["input_ids"]),
                "output_tokens": len(tokens), "token_ids": tokens, "text": "".join(chunks),
                "ttft_seconds": first, "seconds": duration, "decode_tok_s": rate, "events": events,
                "ram_cache": dict(store.stats), "gpu_cache": dict(experts.stats),
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated()}
            if "expected" in case:
                actual = row["text"].strip()
                expected = case["expected"]
                if not isinstance(expected, str):
                    try:
                        actual = json.loads(actual)
                    except ValueError:
                        pass
                passed = actual == expected
                if case["id"] in ("multiply", "remaining", "percentage", "python"):
                    try:
                        passed = Decimal(actual) == Decimal(expected)
                    except (InvalidOperation, TypeError):
                        passed = False
                row.update(passed=passed, expected=expected)
            result["runs"].append(row)
            save()
            print(json.dumps({k: row[k] for k in ("id", "phase", "decode_tok_s", "ttft_seconds", "text")}), flush=True)

        result["status"] = "running"
        with torch.inference_mode():
            if args.mode == "smoke":
                run(fixture["checks"][3], "smoke", 16)
                if not result["runs"][-1]["passed"]:
                    raise RuntimeError("First completion failed the exact extraction check")
            else:
                for case in fixture["checks"]:
                    run(case, "check", 64)
                for phase in ("fresh", "repeat"):
                    for case in fixture["speed"]:
                        run(case, phase, fixture["output_cap"])
                for phase in ("fresh", "repeat"):
                    rates = [r["decode_tok_s"] for r in result["runs"] if r["phase"] == phase]
                    if not rates or any(r is None or r <= 0 for r in rates):
                        raise RuntimeError("Missing usable benchmark measurements")
                    result[phase + "_median_tok_s"] = statistics.median(rates)
        result["status"] = "complete"
    except BaseException as exc:
        result.update(status="failed", error=repr(exc))
        raise
    finally:
        save()
        if model is not None:
            model.unload()
        experts.close()
        store.close()
        print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
