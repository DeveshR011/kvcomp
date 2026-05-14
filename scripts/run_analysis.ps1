$ErrorActionPreference = "Stop"

Set-Location -Path (Split-Path -Parent $PSScriptRoot)

python scripts\analyze_results.py --results-dir results\latest --reports-dir reports\latest

