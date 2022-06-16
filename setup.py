import setuptools
from setuptools import setup
print(setuptools.find_packages("."))
setup(
        name='HolographicLattices',
        version="0.1",
        description="Evaluate nonlinear elliptic PDEs using NR",
        author="Floris Balm",
        author_email="balm@lorentz.leidenuniv.nl",
        package_dir={"":"."},
        packages=setuptools.find_packages(".", exclude=["build", "Manual", "tutorial"]),
        install_requires=['pathos' ,'dill', 'h5py', 'numpy','scipy', 'pyFFTW', 'pyyaml'],
        licence="GPLv3"
        )
