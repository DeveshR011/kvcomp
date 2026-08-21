# KV-Cache Compression Results

Total runs: 945

Scores are substring recall. Intervals are 95% percentile bootstrap.
`oom` counts runs that could not execute at all, which is a different
failure from answering incorrectly and is reported separately.

## By method

| policy | budget | n | score | 95% CI | oom | cache GiB | peak GiB | compress % | decode tok/s |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| full | -1 | 105 | 0.610 | [0.514, 0.705] | 39% | 0.674 | 3.37 | 0% | 7.6 |
| pyramidkv | 1024 | 105 | 0.905 | [0.848, 0.962] | 0% | 0.141 | 3.07 | 80% | 6.4 |
| pyramidkv | 256 | 105 | 0.886 | [0.819, 0.943] | 0% | 0.035 | 2.98 | 95% | 6.0 |
| snapkv | 1024 | 105 | 1.000 | [1.000, 1.000] | 0% | 0.141 | 3.06 | 80% | 6.1 |
| snapkv | 256 | 105 | 0.990 | [0.971, 1.000] | 0% | 0.035 | 2.99 | 95% | 5.8 |
| streaming_llm | 1024 | 105 | 0.267 | [0.181, 0.352] | 0% | 0.141 | 3.06 | 80% | 10.5 |
| streaming_llm | 256 | 105 | 0.171 | [0.105, 0.248] | 0% | 0.035 | 2.99 | 95% | 11.3 |
| tova | 1024 | 105 | 0.943 | [0.895, 0.981] | 0% | 0.141 | 3.06 | 80% | 6.3 |
| tova | 256 | 105 | 0.714 | [0.629, 0.800] | 0% | 0.035 | 2.99 | 95% | 6.3 |

## Score by context length

| policy | budget | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|
| full | -1 | 1.00 | 1.00 | 1.00 | 0.05 (OOM) | 0.00 (OOM) |
| pyramidkv | 1024 | 1.00 | 0.86 | 0.86 | 0.90 | 0.90 |
| pyramidkv | 256 | 1.00 | 0.95 | 0.95 | 0.95 | 0.57 |
| snapkv | 1024 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| snapkv | 256 | 1.00 | 1.00 | 1.00 | 1.00 | 0.95 |
| streaming_llm | 1024 | 0.43 | 0.29 | 0.33 | 0.14 | 0.14 |
| streaming_llm | 256 | 0.29 | 0.14 | 0.14 | 0.14 | 0.14 |
| tova | 1024 | 1.00 | 0.95 | 0.90 | 0.95 | 0.90 |
| tova | 256 | 0.57 | 0.81 | 0.95 | 0.67 | 0.57 |
