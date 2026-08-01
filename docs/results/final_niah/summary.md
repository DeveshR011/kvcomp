# KV-Cache Compression Results

Total runs: 135

Scores are substring recall. Intervals are 95% percentile bootstrap.
`oom` counts runs that could not execute at all, which is a different
failure from answering incorrectly and is reported separately.

## By method

| policy | budget | n | score | 95% CI | oom | cache GiB | peak GiB | compress % | decode tok/s |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| full | -1 | 15 | 0.600 | [0.333, 0.867] | 40% | 0.649 | 3.34 | 0% | 8.9 |
| pyramidkv | 1024 | 15 | 0.333 | [0.133, 0.600] | 0% | 0.141 | 3.07 | 80% | 6.1 |
| pyramidkv | 256 | 15 | 0.733 | [0.533, 0.933] | 0% | 0.035 | 2.98 | 95% | 6.1 |
| snapkv | 1024 | 15 | 1.000 | [1.000, 1.000] | 0% | 0.141 | 3.06 | 80% | 6.1 |
| snapkv | 256 | 15 | 1.000 | [1.000, 1.000] | 0% | 0.035 | 2.99 | 95% | 6.0 |
| streaming_llm | 1024 | 15 | 1.000 | [1.000, 1.000] | 0% | 0.141 | 3.06 | 80% | 10.2 |
| streaming_llm | 256 | 15 | 1.000 | [1.000, 1.000] | 0% | 0.035 | 2.99 | 95% | 11.3 |
| tova | 1024 | 15 | 1.000 | [1.000, 1.000] | 0% | 0.141 | 3.06 | 80% | 6.1 |
| tova | 256 | 15 | 0.667 | [0.400, 0.867] | 0% | 0.035 | 2.99 | 95% | 6.4 |

## Score by context length

| policy | budget | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|
| full | -1 | 1.00 | 1.00 | 1.00 | 0.00 (OOM) | 0.00 (OOM) |
| pyramidkv | 1024 | 1.00 | 0.00 | 0.00 | 0.33 | 0.33 |
| pyramidkv | 256 | 1.00 | 0.67 | 0.67 | 0.67 | 0.67 |
| snapkv | 1024 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| snapkv | 256 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| streaming_llm | 1024 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| streaming_llm | 256 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| tova | 1024 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| tova | 256 | 0.67 | 0.67 | 0.67 | 0.33 | 1.00 |
