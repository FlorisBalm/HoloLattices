# If you set up a virtual env to keep your modules in for this project, activate that first
# source "~/venvs/${PY_HOLO_ENV}/bin/activate"

# Need to have periods for deicmal separator
export LC_NUMERIC=en_US.UTF-8

# Set number of threads used by program for computing matrices etcetera
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# It takes the current BackgroundSolver.py in this folder, and evaluates the example setup

if [ ! -e data ]; then 
    mkdir data;
fi
echo "Running spectral example with sparse matrices"
python3 BackgroundSolver.py --setupfile="spectral_setup.yaml" --log=INFO  --output="data/OutputSpectralSparse.h5" --sparse

echo "Running spectral example with fully dense matrices"
python3 BackgroundSolver.py --setupfile="spectral_setup.yaml" --log=INFO  --output="data/OutputSpectralDense.h5" --dense
