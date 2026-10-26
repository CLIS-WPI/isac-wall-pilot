FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev \
    build-essential git ca-certificates \
    libgl1 libegl1 libopengl0 libatomic1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN python -m pip install --no-cache-dir \
    torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
RUN python -m pip install --no-cache-dir \
    sionna==2.2.0 sionna-rt==2.2.0 \
    numpy scipy pandas pyarrow matplotlib pyyaml \
    pytest nvidia-ml-py

WORKDIR /workspace
COPY . /workspace
RUN python -m pip install --no-cache-dir --no-deps -e .
CMD ["python", "-m", "isac_verify", "--help"]
