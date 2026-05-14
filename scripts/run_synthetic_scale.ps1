$ErrorActionPreference = "Stop"

Set-Location -Path (Split-Path -Parent $PSScriptRoot)

python scripts\generate_synthetic_documents.py
python scripts\run_experiment.py --config config\synthetic_scale_config.json --output-dir results\synthetic_scale --clean-output
python scripts\analyze_results.py --results-dir results\synthetic_scale --questions data\questions\synthetic_scale_questions.json --reports-dir reports\synthetic_scale

