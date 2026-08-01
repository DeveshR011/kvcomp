# Memory-Efficient Long-Context Inference on a 6 GB GPU

Benchmarking training-free **KV-cache compression** for long-context inference
on consumer hardware, evaluated on **RULER** and **Needle-in-a-Haystack**.

**Headline result:** SnapKV retains **5% of the KV cache** while matching
full-context accuracy, and holds that accuracy **flat from 2K to 32K tokens** —
on a card where full context cannot fit past 8K.

```
Qwen3-4B-Instruct (NF4) · RTX 4050 Laptop, 6 GB · 2,169 benchmark runs
```

---

## Why the KV cache is the binding constraint

At long context the weights are a fixed cost; the cache grows linearly and is
what actually ends the run. For Qwen3-4B the arithmetic is unforgiving:

```
2 (K,V) × 36 layers × 8 KV heads × 128 head_dim × 2 bytes = 147,456 B/token
```

| Context | KV cache | Fits beside 2.5 GB of NF4 weights in 6 GB? |
|--------:|---------:|---|
|   4,096 |  0.56 GiB | yes |
|   8,192 |  1.13 GiB | yes |
|  16,384 |  2.25 GiB | no |
|  32,768 |  4.50 GiB | no |
| 131,072 | 18.00 GiB | no |

Compression targets exactly this term, which makes a 6 GB card a legitimate
place to study it rather than a compromise.

---

## Results

### 1. Quality vs. cache size (RULER, 1,080 runs)

Length-matched so the comparison is fair — `full` is averaged only over lengths
it can actually run. n = 120 per policy/budget, 95% bootstrap CIs.

| Policy | Cache | 2K–8K | 16K–32K | Decode tok/s |
|---|---:|---:|---:|---:|
| `pyramidkv` b=1024 | 20% | **0.703** | 0.684 | 7.5 |
| `snapkv` b=1024 | 20% | 0.688 | **0.686** | 9.3 |
| **`snapkv` b=256** | **5%** | **0.677** | **0.684** | **12.0** |
| `tova` b=1024 | 20% | 0.667 | 0.668 | 6.0 |
| `full` | 100% | 0.667 | **0.021** ✗ | 7.3 |
| `pyramidkv` b=256 | 5% | 0.628 | 0.575 | 8.6 |
| `streaming_llm` b=1024 | 20% | 0.110 | 0.016 | 11.5 |
| `h2o` b=1024 | 20% | 0.14 (≤8K) | not run | — |

Three things stand out:

- **SnapKV at 1/20th the cache matches full context** (0.677 vs 0.667) and stays
  flat to 32K, where full context hits a memory wall (0.021 — it cannot fit).
- **Compression decodes faster**: 12.0 vs 7.3 tok/s, since attention scans a
  smaller cache.
- **Score-based selection is what matters.** Position-only StreamingLLM keeps
  the same number of entries and scores 0.045.

### 2. Where each policy goes blind (NIAH, 945 runs)

Score by needle depth at 16K–32K. This is the figure that explains the table above.

| Policy | 0.0 | 0.1 | 0.25 | 0.5 | 0.75 | 0.9 | 1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `snapkv` b=1024 | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `snapkv` b=256 | 1.00 | 1.00 | 1.00 | 0.83 | 1.00 | 1.00 | 1.00 |
| `pyramidkv` b=1024 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.33 |
| `tova` b=1024 | 1.00 | 1.00 | 1.00 | 0.67 | 0.83 | 1.00 | 1.00 |
| `streaming_llm` b=1024 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **1.00** |
| `full` | 0.17 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**StreamingLLM finds the needle only when it sits at the very end** — inside its
recent window. Everywhere else it is blind. Averaged scores hide this; the depth
sweep makes the structural blind spot visible.

At 2K–8K the same policy shows the boundary of its window directly:
`0.00, 0.00, 0.00, 0.00, 0.44, 1.00, 1.00` across increasing depth.

### 3. Per-task (RULER)

| Task | `full` | `snapkv` b=256 | `pyramidkv` b=1024 | `tova` b=1024 | `streaming_llm` |
|---|---:|---:|---:|---:|---:|
| niah_single_1 | 0.65 | **1.00** | **1.00** | **1.00** | 0.00 |
| niah_single_2 | 0.60 | **1.00** | **1.00** | 0.95 | 0.05 |
| niah_multikey_1 | 0.60 | **1.00** | **1.00** | **1.00** | 0.00 |
| niah_multivalue | 0.45 | 0.75 | 0.75 | 0.75 | 0.06 |
| vt (multi-hop) | 0.10 | 0.26 | **0.29** | 0.24 | 0.19 |
| cwe (aggregation) | 0.05 | 0.07 | **0.13** | 0.07 | **0.13** |

Compressed policies **beat full context on needle tasks** (1.00 vs 0.65).
Dropping distractor content appears to help the model focus — consistent with
reports elsewhere in the literature, though n = 20 per task here.

---

## Methods

All training-free, applied during prefill.

| Policy | Evidence used | Scoring cost | Known weakness |
|---|---|---|---|
| `full` | none | none | memory grows without bound |
| `streaming_llm` | position only | none | blind outside sinks + recent window |
| `snapkv` | attention from last `window` queries | O(window × N) | needs tail to proxy the question |
| `pyramidkv` | SnapKV scoring, budget decreasing by depth | O(window × N) | thinner deep layers at small budgets |
| `tova` | attention from the final query only | O(N) | single query is a noisy signal |
| `h2o` | attention accumulated over every query | **O(N²)** | query-agnostic (see Limitations) |

StreamingLLM (arXiv:2309.17453) · H2O (arXiv:2306.14048) ·
SnapKV (arXiv:2404.14469) · TOVA (arXiv:2401.06104) ·
PyramidKV (arXiv:2406.02069) · RULER (arXiv:2404.06654)

---

## Implementation notes

Four decisions that determine whether this works at all.

### Attention scores without monkeypatching

SnapKV and H2O need per-position attention mass, which fused kernels never
materialise. Rather than rewriting attention, `kvcomp/attention.py` registers an
implementation named `kvcomp` in `ALL_ATTENTION_FUNCTIONS`. It computes
attention identically to the stock path and only *additionally* records scores
when asked. A Transformers upgrade cannot silently change the math underneath
the experiment.

### Chunked prefill — compression must happen *during* the read

Compressing once after prefill lowers steady-state memory but leaves the
**peak** untouched: at 32K every policy would first materialise the full 4.50 GiB
cache before evicting anything. Prefill therefore runs in chunks with eviction
after each one, bounding peak cache at `budget + chunk`:

| Prompt | Compress after prefill | Compress per chunk (b=512, chunk=2048) |
|-------:|----------------------:|---------------------------------------:|
|  8,192 | 1.13 GiB | 0.36 GiB |
| 32,768 | 4.50 GiB ✗ | **0.36 GiB** |

Verified output-identical to single-pass prefill.

### Two position spaces

`position_ids` (true text positions, driving RoPE) and `cache_position`
(indices within the cache, driving the causal mask) are tracked separately.
After eviction they diverge. Conflating them corrupts every rotary phase —
silently, since output stays fluent and only quality degrades.

### Query-aware chunking

Query-aware policies rank by what an observation window attends to, and that
window must stand in for the question. With chunked prefill, early chunks are
scored against mid-document filler and discard the answer before the question is
read. Accuracy tracked chunk count exactly:

| Chunks | 1 | 2 | 4 |
|---|---:|---:|---:|
| SnapKV score | 1.00 | 1.00 | **0.00** |

Probing the cache with the prompt's tail before each eviction restores 1.00.

---

## Three memory traps on this hardware

Each surfaced only as an OOM or an unexplained slowdown. All three are
regression-tested.

**1. `enable_gqa` defeats the fused SDPA kernels.** Neither fused kernel accepts
it, so SDPA silently falls back to the MATH backend and materialises the full
attention matrix. At 8,221 tokens:

| Path | Extra memory |
|---|---:|
| `enable_gqa=True` (math fallback) | **8.06 GiB** |
| explicit `repeat_kv` (fused kernel) | **64 MiB** |

A 128× difference, and the sole reason long contexts fit at all. Probe your own
machine with `scripts/check_sdpa_backends.py`.

**2. `lm_head` projects every prompt position** into a 152K vocabulary by
default — ~8 GB at 8K tokens, before the cache matters. Fixed with
`logits_to_keep=1`.

**3. Feasibility must be predicted, not measured.** Windows pages GPU memory to
host RAM instead of failing, so an over-budget run does not error — it crawls at
~50× slower with nothing in the logs. Free-memory readings swing with allocator
state and could not be thresholded reliably; the engine instead predicts total
peak footprint against an absolute ceiling:

```
peak = weights + cache + activations   vs   ceiling = total − external − margin
```

Validated against measurement: 8K predicted 3.84 GiB (measured 3.86, runs);
16K predicted 4.96 GiB (measured 5.20, correctly refused).

---

## Usage

Requires a CUDA-enabled Python with torch, transformers ≥ 5, accelerate,
bitsandbytes. Check the hardware first:

```bash
python scripts/check_sdpa_backends.py 8192
```

Preview a sweep's size, cache cost, and runtime without loading the model:

```bash
python scripts/run_kvcomp.py --config config/kvcomp/final_main.json --dry-run
```

Run and report:

```bash
python scripts/run_kvcomp.py --config config/kvcomp/quick.json        # ~5 min smoke
python scripts/run_kvcomp.py --config config/kvcomp/final_main.json   # RULER, ~3 h
python scripts/run_kvcomp.py --config config/kvcomp/final_niah.json   # NIAH, ~3 h
python scripts/run_kvcomp.py --report results/kvcomp/final_main
```

Results stream to `results.jsonl` per sample and a restart **resumes** rather
than repeats. Run one sweep at a time — two processes on a 6 GB card thrash
rather than fail.

### Tests

```bash
pytest          # 128 CPU tests: policies, benchmarks, scoring, memory model
pytest -m gpu   # end-to-end; needs CUDA and weights
```

The GPU suite covers what unit tests structurally cannot: that compression below
the budget leaves output bit-identical to `full`, that long prefill stays inside
the VRAM budget, that cache bytes match the analytic estimate, and that nothing
leaks across sweep cells.

---

## Reproducibility

| | |
|---|---|
| Model | `Qwen/Qwen3-4B-Instruct-2507`, NF4 double-quantized |
| Hardware | RTX 4050 Laptop, 6.0 GiB VRAM |
| Stack | torch 2.10.0+cu128, transformers 5.2.0, Python 3.14.2 |
| Decoding | greedy, deterministic |
| Runs | 1,080 (RULER) + 144 (H2O) + 945 (NIAH) = **2,169** |
| Scoring | substring recall, per RULER convention |
| Intervals | 95% percentile bootstrap, 2,000 resamples |

All benchmark data is generated from seeded RNG — no downloads, fully offline.
Each (task, length, sample) cell derives its own stream, so adding a task or
length never perturbs existing samples.

Aggregated results are published under [`docs/results/`](docs/results/);
raw per-sample JSONL is gitignored.

---

## Limitations

- **Single model.** Conclusions may not transfer across architectures or
  quantization schemes.
- **Batch size 1** throughout.
- **Compression runs during prefill, not during decoding.** Results describe
  long-prompt, short-generation workloads.
- **H2O underperforms here, and the setting is why.** It ranks by attention
  accumulated over all past queries with no notion of what is being asked.
  Instrumenting the selection showed 949 of 1024 retained positions falling in
  the last three deciles of context. This is not a chunking artefact —
  single-pass prefill scores identically — and normalising the accumulation
  merely inverts the bias from early to late. We keep the published formulation
  and report it as a setting mismatch: H2O's native use is eviction during
  generation, not one-shot prompt compression.
- **`cwe` scores ~0.05–0.13 for every policy including full context.** That is
  the 4B model failing an aggregation task, not a compression result.
- **6 GB caps model size**, so absolute scores are not comparable to leaderboard
  numbers from larger models. The policy-vs-policy comparison under a fixed
  budget is the meaningful axis.

---

## Repository layout

```
kvcomp/            engine, policies, attention capture, analysis
  bench/           RULER + NIAH generators, scoring
config/kvcomp/     sweep configurations
docs/kvcomp.md     design notes and runbook
docs/results/      published aggregate results
docs/ollama_arm.md original prompt-level experiment arm
scripts/           CLI runner, SDPA backend probe
tests/             128 CPU tests + GPU integration suite
```

An earlier Ollama-based arm studying *prompt-level* context strategies
(retrieval, summarisation, sliding windows) also lives in this repository —
see [`docs/ollama_arm.md`](docs/ollama_arm.md). It answers a different question:
prompt-level methods reduce what enters the model, while KV compression reduces
what the model retains after reading everything.
