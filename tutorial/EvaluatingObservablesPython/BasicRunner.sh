# If you set up a virtual env to keep your modules in for this project, activate that first
# source "~/venvs/${PY_HOLO_ENV}/bin/activate"

# Need to have periods for deicmal separator
export LC_NUMERIC=en_US.UTF-8

# Set number of threads used by program for computing matrices etcetera
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# It takes the current BackgroundSolver.py in this folder, and evaluates the example setup

echo "Running basic example"
if [ ! -e data ]; then 
    mkdir data;
fi
python3 ObservablePython.py --log=INFO  --output="data/OutputDataTest.h5" --dense --observables_file Observables.yaml --files data/OutputSpectralDense.h5  data/OutputSpectralDense.h5  data/OutputSpectralDense.h5 


