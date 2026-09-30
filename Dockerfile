# SPDX-License-Identifier: Apache-2.0
# SANKHYA - the CUDA build, the verifier and the demo models in one image (#748).
#
#     docker build -t sankhya .
#     docker run --rm --gpus all sankhya sankhya solve demo/crude_blend.mps --gpu
#     docker run --rm --gpus all sankhya demo/finale.sh
#
# The CUDA runtime and cuSPARSE are linked statically, so the final stage needs no toolkit:
# on a host with the NVIDIA driver and the container toolkit, --gpus all is the whole setup.
# Without a card the same image solves on the CPU and says so.

FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ninja-build cmake g++ git python3 ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN git config --global --add safe.directory /src && \
    cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DSANKHYA_BUILD_TESTS=OFF \
      -DSANKHYA_ENABLE_CUDA=ON -DSANKHYA_CUDA_STATIC=ON \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_DISABLE_FIND_PACKAGE_ZLIB=ON && \
    cmake --build build -j && \
    scripts/package_release.sh build linux-x86_64-cuda /dist && \
    mv /dist/sankhya-*-linux-x86_64-cuda /opt/sankhya

FROM nvidia/cuda:12.6.0-base-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 libgomp1 \
      && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/sankhya /opt/sankhya
WORKDIR /opt/sankhya
ENV PATH=/opt/sankhya/build:$PATH \
    PYTHONPATH=/opt/sankhya/bindings/python
CMD ["sankhya", "version"]
