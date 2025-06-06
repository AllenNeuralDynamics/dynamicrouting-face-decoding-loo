# hash:sha256:053b6952e31367a176fea407fe79cf0b9a0e1279d39ec74a96be33d1736916d1
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG API_KEY
ARG API_SECRET
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U --no-cache-dir \
    aind-data-schema==1.1.1 \
    aind-data-schema-models==0.5.7 \
    code-ocean-aux-tools==1.1.3 \
    hdmf==3.14.6 \
    numba==0.60.0 \
    pandas[all]==2.2.3 \
    polars[plot,deltalake]==1.26.0 \
    pydantic==2.7.4 \
    pynwb==2.8.2 \
    s3fs==2024.10.0 \
    scikit-learn==1.5.2 \
    universal-pathlib==0.1.4 \
    zarr==2.17.2 \
    -e git+https://github.com/AllenInstitute/dynamic_routing_analysis@ad88479ed99947a17967266f32e4a56843ab9c3e#egg=dynamic-routing-analysis
