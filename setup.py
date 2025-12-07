from setuptools import setup, find_packages
import os

def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="slm",
    version="0.1.0",
    description="Superposed Language Model for Music Generation",
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    python_requires='>=3.8',
    include_package_data=True,
)
