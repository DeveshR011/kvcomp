$ErrorActionPreference = "Stop"

Set-Location -Path (Split-Path -Parent $PSScriptRoot)

python scripts\check_environment.py
python scripts\run_experiment.py --config config\safe_6gb_config.json --output-dir results\latest --clean-output
python scripts\analyze_results.py --results-dir results\latest --reports-dir reports\latest
