ARG UBUNTU_RELEASE="18.04"

FROM ubuntu:$UBUNTU_RELEASE as installer
# FROM debian:bullseye-slim as installer

ENV DEBIAN_FRONTEND "noninteractive"

RUN apt-get update && \
    apt-get install -q -y --no-install-recommends \
    apt-utils \
    ca-certificates \
    curl \
    locales \
    unattended-upgrades && \
    unattended-upgrade -d -v

RUN echo "en_GB.UTF-8 UTF-8" >> /etc/locale.gen && \
    locale-gen en_GB.UTF-8 && \
    update-locale LANG=en_GB.UTF-8

WORKDIR /build

ARG CONDA_VERSION="4.10.3"
ARG CONDA_SHA256="1ea2f885b4dbc3098662845560bc64271eb17085387a70c2ba3f29fff6f8d52f"
ENV CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py39_${CONDA_VERSION}-Linux-x86_64.sh"

RUN curl --retry 3 -sSL ${CONDA_URL} > ./miniconda.sh && \
    sha256sum ./miniconda.sh | grep ${CONDA_SHA256} && \
    /bin/sh ./miniconda.sh -b -p /opt/conda && \
    . /opt/conda/etc/profile.d/conda.sh && \
    conda clean -y --all

FROM python:3.9-slim

ARG UNAME="testuser"
ARG UID=1001
ARG GID=1001

RUN apt-get update && \
    apt-get install -q -y --no-install-recommends \
    apt-utils \
    locales \
    unattended-upgrades && \
    unattended-upgrade -d -v && \
    apt-get remove -q -y unattended-upgrades && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN echo "en_GB.UTF-8 UTF-8" >> /etc/locale.gen && \
    locale-gen en_GB.UTF-8 && \
    update-locale LANG=en_GB.UTF-8

RUN groupadd --non-unique --gid ${GID} ${UNAME} && \
    useradd --create-home \
      --non-unique \
      --uid ${UID} \
      --gid ${GID} \
      --shell /bin/bash ${UNAME}

COPY --from=installer --chown=${UID}:${GID} /opt/conda/ /opt/conda/

USER ${UNAME}

COPY --chown=${UID}:${GID} ./tests/docker/condarc /home/${UNAME}/.condarc
COPY --chown=${UID}:${GID}  ./tests/docker/irods_environment.json /home/${UNAME}/.irods/
COPY ./tests/docker/docker-entrypoint.sh /opt/docker/scripts/docker-entrypoint.sh

RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda create -n irods && \
    conda install irods-icommands -n irods && \
    conda install baton -n irods

WORKDIR /code

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY requirements.txt test-requirements.txt /code/

RUN pip install -r requirements.txt -r test-requirements.txt

ENTRYPOINT ["/opt/docker/scripts/docker-entrypoint.sh"]

CMD ["/bin/bash", "-o", "pipefail"]
