FROM nvidia/cuda:11.7.2-devel-ubuntu20.04
ENV PYTHONUNBUFFERED 1

# Setting up basic repo 
ARG DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Setting up working directory
COPY . project/
WORKDIR /project
ENV RUNNING_IN_DOCKER True
ENV CUDA_SUPPORT True
ENV CONDA_DIR "/project/conda"
ENV VENV_DIR "/project/venv"
ENV CONDA_TORCH_CUDA_INSTALLATION = "pytorch[version=2,build=py3.10_cuda11.7*] torchvision torchaudio pytorch-cuda=11.7 cuda-toolkit ninja git -c pytorch -c nvidia/label/cuda-11.7.0 -c nvidia"

# Install prerequisits
RUN apt add-repository -y ppa:deadsnakes/ppa apt-get update && apt-get install -y apt-utils \
    software-properties-common \
    make build-essential wget curl git nano ffmpeg libsm6 libxext6 \
    p7zip-full p7zip-rar \
    git git-lfs\
    python3.10-full python-is-python3 \
    pkg-config libcairo2-dev libjpeg-dev libgif-dev \
    && apt-get clean -y && git lfs install

# Networking
ENV PORT 7860
EXPOSE $PORT

# Setting up environment
RUN /bin/bash /project/install.sh

# Start main runner
CMD ["/bin/bash", "run.sh"]
