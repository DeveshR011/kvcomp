$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))

python scripts\run_experiment.py --config config\budget_sweeps\s256_k1_c100.json --output-dir results\budget_sweeps\s256_k1_c100 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s256_k1_c100 --reports-dir reports\budget_sweeps\s256_k1_c100

python scripts\run_experiment.py --config config\budget_sweeps\s256_k1_c180.json --output-dir results\budget_sweeps\s256_k1_c180 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s256_k1_c180 --reports-dir reports\budget_sweeps\s256_k1_c180

python scripts\run_experiment.py --config config\budget_sweeps\s256_k1_c256.json --output-dir results\budget_sweeps\s256_k1_c256 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s256_k1_c256 --reports-dir reports\budget_sweeps\s256_k1_c256

python scripts\run_experiment.py --config config\budget_sweeps\s256_k2_c100.json --output-dir results\budget_sweeps\s256_k2_c100 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s256_k2_c100 --reports-dir reports\budget_sweeps\s256_k2_c100

python scripts\run_experiment.py --config config\budget_sweeps\s256_k2_c180.json --output-dir results\budget_sweeps\s256_k2_c180 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s256_k2_c180 --reports-dir reports\budget_sweeps\s256_k2_c180

python scripts\run_experiment.py --config config\budget_sweeps\s256_k2_c256.json --output-dir results\budget_sweeps\s256_k2_c256 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s256_k2_c256 --reports-dir reports\budget_sweeps\s256_k2_c256

python scripts\run_experiment.py --config config\budget_sweeps\s256_k3_c100.json --output-dir results\budget_sweeps\s256_k3_c100 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s256_k3_c100 --reports-dir reports\budget_sweeps\s256_k3_c100

python scripts\run_experiment.py --config config\budget_sweeps\s256_k3_c180.json --output-dir results\budget_sweeps\s256_k3_c180 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s256_k3_c180 --reports-dir reports\budget_sweeps\s256_k3_c180

python scripts\run_experiment.py --config config\budget_sweeps\s256_k3_c256.json --output-dir results\budget_sweeps\s256_k3_c256 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s256_k3_c256 --reports-dir reports\budget_sweeps\s256_k3_c256

python scripts\run_experiment.py --config config\budget_sweeps\s256_k4_c100.json --output-dir results\budget_sweeps\s256_k4_c100 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s256_k4_c100 --reports-dir reports\budget_sweeps\s256_k4_c100

python scripts\run_experiment.py --config config\budget_sweeps\s256_k4_c180.json --output-dir results\budget_sweeps\s256_k4_c180 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s256_k4_c180 --reports-dir reports\budget_sweeps\s256_k4_c180

python scripts\run_experiment.py --config config\budget_sweeps\s256_k4_c256.json --output-dir results\budget_sweeps\s256_k4_c256 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s256_k4_c256 --reports-dir reports\budget_sweeps\s256_k4_c256

python scripts\run_experiment.py --config config\budget_sweeps\s512_k1_c100.json --output-dir results\budget_sweeps\s512_k1_c100 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s512_k1_c100 --reports-dir reports\budget_sweeps\s512_k1_c100

python scripts\run_experiment.py --config config\budget_sweeps\s512_k1_c180.json --output-dir results\budget_sweeps\s512_k1_c180 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s512_k1_c180 --reports-dir reports\budget_sweeps\s512_k1_c180

python scripts\run_experiment.py --config config\budget_sweeps\s512_k1_c256.json --output-dir results\budget_sweeps\s512_k1_c256 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s512_k1_c256 --reports-dir reports\budget_sweeps\s512_k1_c256

python scripts\run_experiment.py --config config\budget_sweeps\s512_k2_c100.json --output-dir results\budget_sweeps\s512_k2_c100 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s512_k2_c100 --reports-dir reports\budget_sweeps\s512_k2_c100

python scripts\run_experiment.py --config config\budget_sweeps\s512_k2_c180.json --output-dir results\budget_sweeps\s512_k2_c180 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s512_k2_c180 --reports-dir reports\budget_sweeps\s512_k2_c180

python scripts\run_experiment.py --config config\budget_sweeps\s512_k2_c256.json --output-dir results\budget_sweeps\s512_k2_c256 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s512_k2_c256 --reports-dir reports\budget_sweeps\s512_k2_c256

python scripts\run_experiment.py --config config\budget_sweeps\s512_k3_c100.json --output-dir results\budget_sweeps\s512_k3_c100 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s512_k3_c100 --reports-dir reports\budget_sweeps\s512_k3_c100

python scripts\run_experiment.py --config config\budget_sweeps\s512_k3_c180.json --output-dir results\budget_sweeps\s512_k3_c180 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s512_k3_c180 --reports-dir reports\budget_sweeps\s512_k3_c180

python scripts\run_experiment.py --config config\budget_sweeps\s512_k3_c256.json --output-dir results\budget_sweeps\s512_k3_c256 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s512_k3_c256 --reports-dir reports\budget_sweeps\s512_k3_c256

python scripts\run_experiment.py --config config\budget_sweeps\s512_k4_c100.json --output-dir results\budget_sweeps\s512_k4_c100 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s512_k4_c100 --reports-dir reports\budget_sweeps\s512_k4_c100

python scripts\run_experiment.py --config config\budget_sweeps\s512_k4_c180.json --output-dir results\budget_sweeps\s512_k4_c180 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s512_k4_c180 --reports-dir reports\budget_sweeps\s512_k4_c180

python scripts\run_experiment.py --config config\budget_sweeps\s512_k4_c256.json --output-dir results\budget_sweeps\s512_k4_c256 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s512_k4_c256 --reports-dir reports\budget_sweeps\s512_k4_c256

python scripts\run_experiment.py --config config\budget_sweeps\s768_k1_c100.json --output-dir results\budget_sweeps\s768_k1_c100 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s768_k1_c100 --reports-dir reports\budget_sweeps\s768_k1_c100

python scripts\run_experiment.py --config config\budget_sweeps\s768_k1_c180.json --output-dir results\budget_sweeps\s768_k1_c180 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s768_k1_c180 --reports-dir reports\budget_sweeps\s768_k1_c180

python scripts\run_experiment.py --config config\budget_sweeps\s768_k1_c256.json --output-dir results\budget_sweeps\s768_k1_c256 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s768_k1_c256 --reports-dir reports\budget_sweeps\s768_k1_c256

python scripts\run_experiment.py --config config\budget_sweeps\s768_k2_c100.json --output-dir results\budget_sweeps\s768_k2_c100 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s768_k2_c100 --reports-dir reports\budget_sweeps\s768_k2_c100

python scripts\run_experiment.py --config config\budget_sweeps\s768_k2_c180.json --output-dir results\budget_sweeps\s768_k2_c180 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s768_k2_c180 --reports-dir reports\budget_sweeps\s768_k2_c180

python scripts\run_experiment.py --config config\budget_sweeps\s768_k2_c256.json --output-dir results\budget_sweeps\s768_k2_c256 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s768_k2_c256 --reports-dir reports\budget_sweeps\s768_k2_c256

python scripts\run_experiment.py --config config\budget_sweeps\s768_k3_c100.json --output-dir results\budget_sweeps\s768_k3_c100 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s768_k3_c100 --reports-dir reports\budget_sweeps\s768_k3_c100

python scripts\run_experiment.py --config config\budget_sweeps\s768_k3_c180.json --output-dir results\budget_sweeps\s768_k3_c180 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s768_k3_c180 --reports-dir reports\budget_sweeps\s768_k3_c180

python scripts\run_experiment.py --config config\budget_sweeps\s768_k3_c256.json --output-dir results\budget_sweeps\s768_k3_c256 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s768_k3_c256 --reports-dir reports\budget_sweeps\s768_k3_c256

python scripts\run_experiment.py --config config\budget_sweeps\s768_k4_c100.json --output-dir results\budget_sweeps\s768_k4_c100 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s768_k4_c100 --reports-dir reports\budget_sweeps\s768_k4_c100

python scripts\run_experiment.py --config config\budget_sweeps\s768_k4_c180.json --output-dir results\budget_sweeps\s768_k4_c180 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s768_k4_c180 --reports-dir reports\budget_sweeps\s768_k4_c180

python scripts\run_experiment.py --config config\budget_sweeps\s768_k4_c256.json --output-dir results\budget_sweeps\s768_k4_c256 --clean-output --max-questions 2
python scripts\analyze_results.py --results-dir results\budget_sweeps\s768_k4_c256 --reports-dir reports\budget_sweeps\s768_k4_c256
