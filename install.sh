#!/bin/bash
if [[ "$CUDA_SUPPORT" = "True" ]]
then
    printf "\n%s\n" "${delimiter}"
    printf "Creating conda environment"
    printf "\n%s\n" "${delimiter}"
    # Download and install miniconda
     curl -Lk "https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh" > "miniconda_installer.sh" \
        && chmod u+x "miniconda_installer.sh" \
        && /bin/bash "miniconda_installer.sh" -b -p "$CONDA_DIR" \
        && echo "Installed miniconda version:" \
        && "${CONDA_DIR}/bin/conda" --version

    # Create conda environment
    "${CONDA_DIR}/bin/conda" create -y -k --prefix "$VENV_DIR" python=3.10
    source "${CONDA_DIR}/etc/profile.d/conda.sh"
    conda activate "$VENV_DIR"
    conda install -y -k  $CONDA_TORCH_CUDA_INSTALLATION
else
    printf "\n%s\n" "${delimiter}"
    printf "Creating virtualenv environment"
    printf "\n%s\n" "${delimiter}"
    python3.10 -m "$VENV_DIR" venv_cpu
    source "${VENV_DIR}/bin/activate"
fi
printf "\n%s\n" "${delimiter}"
printf "Installing pip requirements"
printf "\n%s\n" "${delimiter}"

python -m pip install -r requirements.txt

printf "\n%s\n" "${delimiter}"
printf "Finished installation"
printf "\n%s\n" "${delimiter}"

exit 0
