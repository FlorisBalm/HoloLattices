export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

set -euo pipefail

output_file="data/StokesOutput.txt"
backgrounds=( ./NoSSB_B0_P1_T0.01_A0.5.h5 )


#If pythonpath doesn't exist yet, this will not throw an error
export PYTHONPATH="${PYTHONPATH:-}:."

python3 -W ignore ./StokesFlowComputation.py --setup_replace="StokesBGReplace.yaml" --log=DEBUG --output_file "${output_file}"  --background "${backgrounds[@]}"  --overwrite --sparse
