$ErrorActionPreference = "Stop"

Set-Location -Path (Split-Path -Parent $PSScriptRoot)

python scripts\check_environment.py
python scripts\run_experiment.py --config config\safe_6gb_config.json --methods full_context retrieval_memory_tfidf retrieval_plus_summary adaptive_context --output-dir results\adaptive_ablation --clean-output
python scripts\analyze_results.py --results-dir results\adaptive_ablation --reports-dir reports\adaptive_ablation

