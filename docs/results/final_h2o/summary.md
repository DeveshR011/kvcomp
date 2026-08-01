# KV-Cache Compression Results

Total runs: 144

Scores are substring recall. Intervals are 95% percentile bootstrap.
`oom` counts runs that could not execute at all, which is a different
failure from answering incorrectly and is reported separately.

## By method

| policy | budget | n | score | 95% CI | oom | cache GiB | peak GiB | compress % | decode tok/s |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| h2o | 1024 | 72 | 0.144 | [0.082, 0.212] | 0% | 0.141 | 3.06 | 70% | 12.9 |
| h2o | 256 | 72 | 0.065 | [0.027, 0.106] | 0% | 0.035 | 2.98 | 93% | 12.7 |

## Score by context length

| policy | budget | 2048 | 4096 | 8192 |
|---|---:|---:|---:|---:|
| h2o | 1024 | 0.28 | 0.09 | 0.06 |
| h2o | 256 | 0.10 | 0.02 | 0.07 |
