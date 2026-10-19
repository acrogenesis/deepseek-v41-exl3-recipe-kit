# 256K context measurements — 2026-10-19

The final result is **29.939670 tok/s repeated / 9.512925 tok/s fresh** with
262,015 input tokens and 128 output tokens. See [the report and command](../../CONTEXT256K.md).

The final run is `filled-desktop.json`; `filled-desktop-audit.json` independently
recomputes rates from its output events and validates full-window evidence.
`filled-01.json` is the earlier full-window baseline. Both phases produce the
same 128 tokens in both configurations. The fixture is `filled-fixture.json`.
Each request resets KV and recurrent state; expert and Engram caches can remain
warm. Timings exclude the first output token. This is one request per phase,
not a statistical estimate across many prompts.

The final configuration uses 80 GiB pinned expert RAM, 18 GiB expert GPU space
during prefill and 22.5 GiB during decode. Other fast flags, exact checkpoint
revisions and source digests are recorded in the result. GPU memory and model
precision are not inferred from a download size. All 59 original source files
were rehashed before this campaign; `source-reverification.json` records that
check. The tested runtime is recorded in `desktop-snapshot-sources.json`.
`committed-source-equivalence.json` records one final formatting-only change:
removal of a trailing empty line, with identical Python AST. All other runtime
files are byte-identical to the tested snapshot; all 31 CPU tests pass afterward.

## Evidence map

- `filled-desktop.json`: actual filled-window timings, output tokens, retrieval,
  native indexer comparisons, final KV positions, resource and cache counters.
- `filled-desktop-audit.json`: successful final audit; compares all 256 output
  tokens with the baseline and confirms every native comparison occurs at the
  end of the full prompt. Its speed result is separate from its validity flag.
- `filled-desktop-routes.json`: observed generation working set and real cache
  hit rates. The additional retrospective static-cache estimate is diagnostic;
  it is not achieved performance and was never fed back as a future-route oracle.
- `filled-desktop-memory.jsonl` and `-memory-summary.json`: 30-second external
  samples, beginning partway through fresh prefill. Both model/controller and
  monitor terminated successfully. Samples are not a continuous minimum.
- `indexer-gate-04.json`: 18 synthetic selection/mask comparisons through 262K.
- `elastic-priority-v2-gate.json`: real K2/K3 byte and asynchronous lifetime
  checks, including LFU eviction with untrained prefill frequencies.
- `cpu-tests-desktop-budget.txt`: 31 passing CPU tests on the final runtime.
- `audit-reject-short-context.json`: negative control demonstrating that an
  8K prompt with a 256K allocation does not establish filled-window performance.

## Screening runs and failures

`short-allocation.json` keeps the old short prompts and only enlarges capacity.
Its 29.82 repeated tok/s is not a filled-window measurement. `prefill-*`,
`elastic-8192.json`, `priority-v2-8k.json`, `engram-priority-v2-8k.json` and
`desktop-8k.json` are shorter screening runs, clearly identified by input length.
The protected-cache screening runs preserve every token of the earlier 8K run.

`filled-priority-v2.json` stopped at 122,880 inputs when available host RAM fell
below the 12 GiB reserve. `ram80-8k.json` subsequently hit CUDA OOM with a 23 GiB
GPU cache as desktop VRAM usage increased. Neither failure contributes a speed
result. The final 80 GiB RAM / 22.5 GiB GPU run completed both full-window phases
with no model swap or OOM. Initial 23 GiB / 2K-prefill allocation failures and
the original bare-JSON scorer's corrected fenced-JSON retrieval are also retained.

Archived queue states and logs describe experimental orchestration at the times
shown. They are not current process-liveness evidence. Recipe GPU jobs ran
serially; desktop applications remained open.
