#!/usr/bin/env bash
# Sets up the conda environment for this project.
# Requirements: conda must be installed (https://docs.anaconda.com/miniconda/)
# Usage: bash setup_env.sh

set -euo pipefail

ENV_NAME="proyecto"
PYTHON_VERSION="3.12"
FAISS_VERSION="1.14.1"

echo "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

echo "Installing faiss-gpu via conda..."
conda install -n "${ENV_NAME}" \
    -c pytorch -c nvidia -c conda-forge \
    faiss-gpu="${FAISS_VERSION}" -y

echo "Installing remaining dependencies via pip..."
conda run -n "${ENV_NAME}" pip install -r requirements.txt

echo ""
echo "Done. Activate the environment with:"
echo "    conda activate ${ENV_NAME}"
