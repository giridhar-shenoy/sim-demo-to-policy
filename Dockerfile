# CI and devcontainer image for sim-demo-to-policy (Python-only repo, no ROS).
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    SDP_MENAGERIE_DIR=/opt/mujoco_menagerie

# libegl-mesa0 + libgl1-mesa-dri provide the Mesa EGL driver and llvmpipe that
# MUJOCO_GL=egl needs without a GPU; libosmesa6 is the fallback backend.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgl1 libegl1 libegl-mesa0 libgl1-mesa-dri libosmesa6 ffmpeg git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Franka Panda model from mujoco_menagerie at a pinned commit (fetched, not vendored).
COPY scripts/fetch_menagerie.sh /tmp/fetch_menagerie.sh
RUN bash /tmp/fetch_menagerie.sh "$SDP_MENAGERIE_DIR" && rm /tmp/fetch_menagerie.sh

# Exact dependency versions from the laptop venv. torch comes from the CPU index
# so the CUDA build is never pulled.
COPY requirements.lock /tmp/requirements.lock
RUN pip install -r /tmp/requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu

RUN useradd --create-home --uid 1000 dev
WORKDIR /workspace
COPY --chown=dev:dev . .
RUN pip install --no-deps -e . && chown -R dev:dev /workspace
USER dev
CMD ["bash"]
