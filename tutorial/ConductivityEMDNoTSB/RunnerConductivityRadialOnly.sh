export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
set -euo pipefail
output_file="data/ConductivityOutput.txt"
backgrounds=( ExampleEMDBackground.h5)

python3 -W ignore ./ConductivityOnBackground.py --setup_replace="SettingsBGReplace.yaml" --log=DEBUG --output_file "${output_file}"  --background "${backgrounds[@]}"  --overwrite --sparse  # --no_use_pathos
