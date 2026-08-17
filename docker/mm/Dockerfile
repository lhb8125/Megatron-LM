# Overlay the current Megatron-LM working tree onto the prebuilt sync-free image.
#
# The base image (built from examples/sync_free_hybridep/Dockerfile*) already ships
# CUDA/cuDNN/cuBLAS + PyTorch + TransformerEngine + DeepEP(hybrid-ep) + resiliency-ext.
# This stage just bakes the repo's code into it so a container runs the local
# changes without a bind-mount.
#
# Build via docker/mm/build_docker.sh (sets BASE_IMAGE / tag). Build context is
# the repo root; large local artifacts are excluded by /.dockerignore.

ARG BASE_IMAGE=megatron-sync-free-grouped-tensor:b300
FROM ${BASE_IMAGE}

# Install py-spy so the platform's hang detector can dump per-rank Python/native
# stacks (py-spy dump --native) when a job hangs.
RUN pip install --no-cache-dir py-spy

# Bake the current working tree into the image at the path the launcher uses.
RUN rm -rf /workspace/Megatron-LM
WORKDIR /workspace/Megatron-LM
COPY . /workspace/Megatron-LM/
ENV PYTHONPATH=/workspace/Megatron-LM:${PYTHONPATH}

# Optionally pre-compile the C++ dataset helpers so the first job doesn't pay the
# runtime `make`. Off by default; enable with --build-arg BAKE_HELPERS=1.
# Rebuilds when helpers.cpp is newer than the built .so.
ARG BAKE_HELPERS=0
RUN if [ "$BAKE_HELPERS" = "1" ]; then \
        cd /workspace/Megatron-LM/megatron/core/datasets && \
        SUFFIX=$(python3-config --extension-suffix) && \
        if [ ! -f "helpers_cpp${SUFFIX}" ] || [ "helpers_cpp${SUFFIX}" -ot helpers.cpp ]; then \
            make helpers_cpp${SUFFIX} && ls -la helpers_cpp*.so ; \
        else \
            echo "helpers_cpp${SUFFIX} up-to-date; skipping" ; \
        fi ; \
    else \
        echo "BAKE_HELPERS not set; runtime compile_helpers() will build them on first use" ; \
    fi
