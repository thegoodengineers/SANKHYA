# SPDX-License-Identifier: Apache-2.0
# The board's backend: a Linux build of the solver plus the FastAPI layer.
# Build from the REPOSITORY ROOT so the solver sources are in the context:
#     docker build -f apps/web/Dockerfile -t sankhya-board .
#     docker run -p 8010:8010 sankhya-board

FROM gcc:13 AS build
RUN apt-get update && apt-get install -y --no-install-recommends cmake ninja-build zlib1g-dev git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DSANKHYA_BUILD_TESTS=OFF \
    && cmake --build build -j --target sankhya

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends zlib1g && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir fastapi uvicorn python-multipart highspy
WORKDIR /app
COPY --from=build /src/build/sankhya /app/build/sankhya
COPY apps/web /app/apps/web
COPY tools /app/tools
COPY demo /app/demo
ENV HOST=0.0.0.0 PORT=8010
EXPOSE 8010
CMD ["python", "apps/web/server.py"]
