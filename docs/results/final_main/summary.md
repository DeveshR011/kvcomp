# KV-Cache Compression Results

Total runs: 1080

Scores are substring recall. Intervals are 95% percentile bootstrap.
`oom` counts runs that could not execute at all, which is a different
failure from answering incorrectly and is reported separately.

## By method

| policy | budget | n | score | 95% CI | oom | cache GiB | peak GiB | compress % | decode tok/s |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| full | -1 | 120 | 0.408 | [0.331, 0.490] | 38% | 0.691 | 3.39 | 0% | 7.3 |
| pyramidkv | 1024 | 120 | 0.695 | [0.623, 0.760] | 0% | 0.141 | 3.07 | 80% | 7.5 |
| pyramidkv | 256 | 120 | 0.607 | [0.529, 0.681] | 0% | 0.035 | 2.98 | 95% | 8.6 |
| snapkv | 1024 | 120 | 0.687 | [0.614, 0.751] | 0% | 0.141 | 3.06 | 80% | 9.3 |
| snapkv | 256 | 120 | 0.680 | [0.608, 0.748] | 0% | 0.035 | 2.99 | 95% | 12.0 |
| streaming_llm | 1024 | 120 | 0.072 | [0.043, 0.103] | 0% | 0.141 | 3.06 | 80% | 11.5 |
| streaming_llm | 256 | 120 | 0.045 | [0.022, 0.072] | 0% | 0.035 | 2.99 | 95% | 8.7 |
| tova | 1024 | 120 | 0.667 | [0.592, 0.735] | 0% | 0.141 | 3.06 | 80% | 6.0 |
| tova | 256 | 120 | 0.603 | [0.526, 0.679] | 0% | 0.035 | 2.99 | 95% | 8.2 |

## Score by context length

| policy | budget | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|
| full | -1 | 0.70 | 0.66 | 0.65 | 0.04 (OOM) | 0.00 (OOM) |
| pyramidkv | 1024 | 0.72 | 0.65 | 0.74 | 0.69 | 0.68 |
| pyramidkv | 256 | 0.66 | 0.62 | 0.61 | 0.62 | 0.53 |
| snapkv | 1024 | 0.70 | 0.67 | 0.69 | 0.70 | 0.67 |
| snapkv | 256 | 0.66 | 0.65 | 0.72 | 0.72 | 0.65 |
| streaming_llm | 1024 | 0.20 | 0.09 | 0.04 | 0.02 | 0.01 |
| streaming_llm | 256 | 0.06 | 0.03 | 0.04 | 0.02 | 0.07 |
| tova | 1024 | 0.70 | 0.66 | 0.65 | 0.69 | 0.65 |
| tova | 256 | 0.65 | 0.65 | 0.64 | 0.65 | 0.44 |
