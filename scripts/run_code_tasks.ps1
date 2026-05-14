$ErrorActionPreference = "Stop"

Set-Location -Path (Split-Path -Parent $PSScriptRoot)

python scripts\check_environment.py
python scripts\run_experiment.py --config config\code_tasks_config.json --output-dir results\code_tasks --clean-output
python scripts\analyze_results.py --results-dir results\code_tasks --reports-dir reports\code_tasks
