$ErrorActionPreference = "Stop"

Set-Location -Path (Split-Path -Parent $PSScriptRoot)

python scripts\generate_budget_sweeps.py --max-questions 2
Write-Host "Generated budget sweep configs. Review config\budget_sweeps before running the generated run_budget_sweeps.ps1."

