# KV-Cache Compression under a 6 GB VRAM Budget

A benchmark for training-free KV-cache compression on consumer hardware,
evaluated on RULER and Needle-in-a-Haystack.

## What this measures

Long-context inference on a small GPU is bounded by the KV cache, not by the
weights. For Qwen3-4B the arithmetic is fixed and unforgiving:

```
2 (K and V) x 36 layers x 8 KV heads x 128 head_dim x 2 bytes = 147,456 B/token
```

| Context | KV cache | Fits beside 2.5 GB of NF4 weights on a 6 GB card? |
|--------:|---------:|---|
|   4,096 |  0.56 GiB | yes |
|   8,192 |  1.13 GiB | yes |
|  16,384 |  2.25 GiB | tight |
|  32,768 |  4.50 GiB | **no** |
| 131,072 | 18.00 GiB | **no** |

The weights are a fixed cost; the cache grows linearly with context and is what
actually ends the run. Compression targets exactly this term, which is why a
6 GB card is a legitimate place to study it rather than a compromise.

## Methods

All are training-free and applied after prefill.

| Policy | Evidence used | Scoring cost | Known weakness |
|---|---|---|---|
| `full` | none | none | memory grows without bound |
| `streaming_llm` | position only | none | blind to the middle of the context |
| `snapkv` | attention from the last `window` queries | O(window x N) | needs the tail to be a good proxy for the question |
| `pyramidkv` | same as SnapKV, budget decreasing by depth | O(window x N) | as SnapKV |
| `tova` | attention from the final query only | O(N) | a single query is a noisy ranking signal |
| `h2o` | attention accumulated over every query | **O(N²)** | early positions accumulate mass simply by being early |

References: StreamingLLM (arXiv:2309.17453), H2O (arXiv:2306.14048),
SnapKV (arXiv:2404.14469), TOVA (arXiv:2401.06104), PyramidKV (arXiv:2406.02069).

## Design

### Attention scores without monkeypatching

SnapKV and H2O need per-position attention mass, which fused kernels never
materialise. Instead of rewriting the model's attention, `kvcomp/attention.py`
registers an implementation named `kvcomp` in `ALL_ATTENTION_FUNCTIONS`.
Transformers resolves it by name, so selecting it is one argument at load time.
It computes attention identically to the stock path and, only when a capture
request is attached, additionally computes the requested scores. Nothing in the
modelling code is patched, so a Transformers upgrade cannot silently change the
attention math underneath the experiment.

### Positions after eviction

Compressing the cache breaks an assumption in `model.generate`: that a cached
entry's index equals its position in the text. After eviction the cache may hold
1,024 entries from a 32k prompt, so `get_seq_length()` returns 1,024 while the
next token belongs at position 32,000. `generate` derives RoPE positions from
cache length and would place it at 1,024, corrupting every rotary phase.

The damage is invisible: output stays fluent, and the resulting quality drop
looks like the compression method failing rather than a positional bug. So
`kvcomp/engine.py` runs its own decode loop that tracks the true text position
independently of cache occupancy. Cached keys retain the rotary phases they were
built with, as the published methods specify.

### Chunked prefill: why compression must happen *during* the read

Compressing once after prefill lowers steady-state memory but leaves the **peak**
untouched. At 32k every policy — SnapKV included — would first materialise the
full 4.50 GiB cache before it could evict anything, so 32k was infeasible for
every method alike and compression's entire benefit went unrealised.

The engine therefore prefills in chunks of `prefill_chunk` tokens and compresses
after each one. Peak cache becomes `budget + prefill_chunk` rather than the whole
prompt:

| Prompt | Compress after prefill | Compress per chunk (budget 512, chunk 2048) |
|-------:|----------------------:|--------------------------------------------:|
|  8,192 | 1.13 GiB | 0.36 GiB |
| 32,768 | 4.50 GiB (infeasible) | 0.36 GiB |

Chunking is verified to be output-identical to a single-pass prefill; the two
paths agree token for token on the same prompt.

Two position spaces are kept distinct throughout, which is what makes this safe:

- `position_ids` — true text positions, driving RoPE.
- `cache_position` — indices *within the cache*, driving the causal mask.

After eviction these diverge. Using text positions for masking would let a
chunk's tokens attend to each other non-causally; using cache indices for RoPE
would corrupt every rotary phase.

### Two memory traps this hardware exposes

Both were found by measurement, and both present as an OOM with no other symptom.

**1. `enable_gqa` defeats the fused kernels.** PyTorch's SDPA accepts
`enable_gqa=True` to handle grouped-query attention without expanding KV heads.
On this build neither fused kernel supports it — *"both fused kernels require
query, key and value to have the same num_heads"* — so SDPA falls back to the
MATH backend and materialises the full `[heads, q, kv]` attention matrix.
Measured at 8,221 tokens:

| Path | Extra memory |
|---|---:|
| `enable_gqa=True` (math fallback) | **8.06 GiB** |
| explicit `repeat_kv` (fused kernel) | **64 MiB** |

A 128x difference, and the single reason long contexts fit here at all. The
engine expands KV heads explicitly.

**2. Logits are computed for every prompt position.** `lm_head` projects the
full sequence into a 152k-entry vocabulary by default: ~8 GB at 8k tokens,
before the cache is even a factor. The engine passes `logits_to_keep=1`, since
only the final position drives generation.

An unregistered attention name has a third, quieter version of the same
problem: the mask builder falls back to a dense eager mask. The engine registers
`kvcomp` against SDPA's mask builder so the causal fast path is preserved.

## Benchmarks

**RULER** (arXiv:2404.06654) — synthetic, seeded, no downloads. Three task
shapes, chosen because compression fails differently on each:

- *Needle* (`niah_*`) — reward keeping a few specific positions.
- *Aggregation* (`cwe`, `fwe`) — evidence is spread across the whole context, so
  no small subset suffices. This is where sink+recent heuristics should break.
- *Multi-hop* (`vt`) — needs a chain of positions, punishing policies that keep
  isolated high-scoring tokens without their context.

A suite of only single-needle tasks would flatter every method; the aggregation
tasks are what make the comparison informative.

**NIAH** — sweeps needle depth against context length. Averaging over depth
would hide a mid-context blind spot behind a mediocre mean; the depth x length
grid shows it directly.

Scoring is substring recall, following RULER. Generative models pad answers with
commentary, so exact match would understate real accuracy.

## Runbook

Use the CUDA-enabled interpreter. On this machine that is the 3.14 build:

```bash
C:/Users/rawat/AppData/Local/Python/pythoncore-3.14-64/python.exe scripts/run_kvcomp.py --config config/kvcomp/pilot.json
```

Check the hardware first — this reports whether SDPA has a working fused kernel,
which decides whether long context is viable at all:

```bash
python scripts/check_sdpa_backends.py 8192
```

Preview a sweep's size and predicted cache cost without loading the model:

```bash
python scripts/run_kvcomp.py --config config/kvcomp/ruler_6gb.json --dry-run
```

### Which sweep to run

| Config | Runs | Max context | Estimated | Purpose |
|---|---:|---:|---|---|
| `quick` | 48 | 2K | ~5 min | Sanity check; all policies, all task shapes |
| `pilot` | 252 | 8K | ~25 min | Directional results before committing |
| `ruler_main` | 1,080 | 32K | ~3.5 h | The headline sweep |
| `ruler_h2o` | 144 | 8K | ~30 min | H2O, run separately (see below) |
| `ruler_6gb` | 2,880 | 32K | ~19 h | Exhaustive; 3 budgets, 6 samples |

```bash
python scripts/run_kvcomp.py --config config/kvcomp/quick.json
python scripts/run_kvcomp.py --config config/kvcomp/ruler_main.json
python scripts/run_kvcomp.py --config config/kvcomp/ruler_h2o.json
```

`ruler_main` + `ruler_h2o` together cover every policy in about **4 hours**.

**Why H2O is a separate config.** Its scoring is quadratic in context length,
so at 32k a single H2O run costs minutes rather than seconds. Left in the main
sweep it accounted for 11.5 of 19.2 hours — 60% of the total for one of six
policies. Capping it at 8k keeps the comparison meaningful where it is
affordable, and the cost itself is a reportable finding rather than a nuisance.

`--dry-run` prints a runtime estimate and a per-policy breakdown before anything
loads. Check it before launching; it is calibrated at 8k and extrapolated, so
treat it as an order of magnitude rather than a promise.

Build reports at any time, including while a sweep is still running:

```bash
python scripts/run_kvcomp.py --report results/kvcomp/ruler_6gb
```

**Run only one sweep at a time.** Two processes sharing a 6 GB card will thrash
rather than fail, and both append to the same results file. Verify with
`nvidia-smi` before launching.

Results stream to `results.jsonl` per sample and a restart resumes rather than
repeating, so interrupting a sweep is safe.

### Cost

Measured on an RTX 4050 (6 GB) with Qwen3-4B at NF4:

| Prompt | Policy | Wall time |
|-------:|---|---:|
|  8,192 | snapkv | ~8 s |
|  8,192 | h2o | ~23 s (quadratic scoring) |
| 14,521 | full | ~177 s |

Cost grows sharply with context because attention is quadratic and a large cache
slows every decode step. Budget accordingly: the 32k cells dominate any sweep
that includes them.

Results stream to `results.jsonl` as each sample completes, and a restart
resumes rather than repeating. On this hardware a sweep runs for hours and one
oversized cell can hard-OOM the process, so both behaviours are correctness
features rather than conveniences.

Reports carry 95% bootstrap confidence intervals. With a few samples per cell a
3-point accuracy gap is usually noise, and scores are bounded and often bimodal
(a needle is found or it is not), so a normal approximation would misstate the
uncertainty. OOM rate is reported separately from accuracy: a method that could
not run has not "scored zero" in the same sense as one that ran and was wrong.

## Tests

```bash
pytest              # CPU: policies, benchmark generation, scoring
pytest -m gpu       # adds end-to-end runs; needs CUDA and weights
```

The GPU suite covers what unit tests structurally cannot: that compression below
the budget leaves output bit-identical to `full`, that long prefill stays inside
the VRAM budget, that cache bytes match the analytic estimate, and that nothing
leaks across sweep cells.

## Limitations

- Single model (Qwen3-4B at NF4). Conclusions may not transfer across
  architectures or quantization schemes.
- Batch size 1 throughout.
- Compression runs during prefill and once more at its end, but not during
  decoding. Results therefore describe long-prompt, short-generation workloads.
- **H2O is query-agnostic, and this benchmark is query-conditioned.** H2O ranks
  by attention accumulated over every past query, with no notion of what is
  being asked. On RULER's needle tasks it scored 0.00 while SnapKV scored 1.00,
  and instrumenting the selection showed why: of 1024 retained positions, 949
  fell in the last three deciles of the context and only 2 in the needle's
  decile.

  This is not a chunking artefact -- single-pass prefill scores the same -- and
  it is not fixable by normalising the accumulation. Dividing by the number of
  attending queries removes the early-position bias but introduces a late-
  position one, because trailing positions are seen by few, adjacent queries
  whose attention is locally high. Retention moved from the start of the
  document to its end and the score stayed at 0.00.

  We therefore keep the published cumulative-sum formulation and report the
  result as a finding: accumulated-attention eviction is a poor fit for
  compressing a prompt whose question sits at the end, which is the setting
  window-based methods such as SnapKV were designed for. H2O's native setting is
  eviction during generation, not one-shot prompt compression.
- Query-aware policies probe the cache with the prompt's tail before each
  eviction (see `EngineConfig.query_aware_chunks`). Without it, chunked prefill
  discards the answer before the question is read: accuracy tracked chunk count
  exactly, 1.00 / 1.00 / 0.00 for one / two / four chunks.
- The feasibility precheck estimates activation memory with a coarse constant.
  It is tuned for this model on a 6 GB card and would need re-checking
  elsewhere; when it is wrong it is optimistic, and an over-budget run degrades
  into host-memory spilling rather than a clean error.
- H2O's O(N²) scoring makes it the slowest policy by a wide margin; its prefill
  cost is a measured result here, not an implementation defect.
- 6 GB caps the model size, so absolute scores are not comparable to
  leaderboard numbers from larger models. The comparison between policies under
  a fixed budget is the meaningful axis.
