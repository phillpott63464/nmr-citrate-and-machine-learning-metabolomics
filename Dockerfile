# FROM ubuntu:22.04
# FROM rocm/pytorch
# FROM rocm/dev-ubuntu-22.04
FROM rocm/tensorflow

# Prevent interactive prompts
# ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies first
RUN apt-get update && apt-get install -y wget gnupg2 curl software-properties-common && rm -rf /var/lib/apt/lists/*

# Add AMD's ROCm APT repo (key + source)
RUN wget -qO - http://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - && \
    echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/6.0 ubuntu main' | tee /etc/apt/sources.list.d/rocm.list

RUN apt-get update && apt-get install -y \
    rocm-core \
    rocm-hip-runtime \
    rocm-dev \
    && rm -rf /var/lib/apt/lists/*

# Update again *after* adding repo
RUN apt-get update && apt-get install -y \
    libhsa-runtime64-1 \
    rocm-hip-libraries \
    # rocm-smi \
    hip-runtime-amd \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    conda clean -afy

COPY app/environment.yml /app/environment.yml
COPY app/requirements.txt /app/requirements.txt
# Accept the ToS for required channels
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda env create -f /app/environment.yml && conda clean -afy
# RUN $CONDA_DIR/envs/ml/bin/pip install tensorflow-rocm==2.14.0.600 -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.0/ --upgrade
RUN $CONDA_DIR/envs/ml/bin/pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.0/


WORKDIR /usr/src/app

SHELL ["conda", "run", "-n", "ml", "/bin/bash", "-c"]

CMD ["conda", "run", "-n", "ml", "marimo", "edit", "--host", "0.0.0.0", "--port", "2718", "--no-token"]