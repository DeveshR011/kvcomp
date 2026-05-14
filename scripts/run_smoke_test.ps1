$ErrorActionPreference = "Stop"

Set-Location -Path (Split-Path -Parent $PSScriptRoot)

python scripts\check_environment.py
python scripts\run_experiment.py --config config\safe_6gb_config.json --methods retrieval_memory_tfidf --max-questions 1 --repeat-runs 1 --warmup-runs 0 --output-dir results\smoke --clean-output
python scripts\analyze_results.py --results-dir results\smoke --reports-dir reports\smoke
