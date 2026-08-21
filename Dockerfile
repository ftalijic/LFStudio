# syntax=docker/dockerfile:1
#
# LichtFeld Studio, baked into a reusable RunPod image.
#
# Why two stages: stage 1 does the ~60-90 min compile (GCC-14 + CUDA 12.8 +
# vcpkg + LichtFeld's own build). That only runs when this image is rebuilt
# (new LFS_REF, or you change this file) - GitHub Actions does it on its own
# runners, not on a paid RunPod GPU. Stage 2 is the slim image RunPod
# actually pulls: just the compiled binary + the runtime libs it needs,
# discovered via `ldd` rather than guessed, so we don't ship a devel
# toolchain (gcc, vcpkg, build headers) on every pod boot.
#
# SSH setup below mirrors oblaQ/Git/files/Dockerfile + start.sh (the
# COLMAP/Nerfstudio image) - RunPod's stock templates bundle SSH access
# automatically, but a custom image does not. That combo was already
# debugged once (a first deploy hit "Connection refused" until
# openssh-server + the $PUBLIC_KEY-at-container-start dance were added), so
# this reuses the same proven pattern instead of re-discovering it.

########################################
# Stage 1: builder
########################################
ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS builder

# Which LichtFeld-Studio git ref to build. Override at build time with
# --build-arg LFS_REF=v0.5.2 (or a commit SHA) to pin an exact version
# instead of always tracking master.
ARG LFS_REF=master
ENV DEBIAN_FRONTEND=noninteractive

# GCC 14 installs directly via apt on Ubuntu 24.04+ (per the project's own
# Linux build wiki - no PPA needed here, unlike older Ubuntu releases).
# The X11/GL/GTK -dev packages are needed at BUILD/link time even for a
# headless CLI run, since the GUI code paths are compiled into the same
# binary regardless of the --headless runtime flag.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates gnupg wget curl git zip unzip tar pkg-config \
        build-essential \
        gcc-14 g++-14 gfortran-14 \
        ninja-build nasm python3 python3-pip \
        autoconf autoconf-archive automake libtool \
        libglu1-mesa-dev libgtk-3-dev xorg-dev libgl1-mesa-dev libegl1-mesa-dev \
        libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 100 \
    && rm -rf /var/lib/apt/lists/*

# Modern CMake from Kitware directly - mirrors what LichtFeld's own
# docker/Dockerfile does (it explicitly pulls CMake 4.0.3 rather than
# trusting Ubuntu 24.04's packaged 3.28, which is a signal their
# CMakeLists.txt needs newer than that).
RUN wget -qO- https://apt.kitware.com/keys/kitware-archive-latest.asc \
        | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ noble main" \
        > /etc/apt/sources.list.d/kitware.list \
    && apt-get update && apt-get install -y --no-install-recommends cmake \
    && rm -rf /var/lib/apt/lists/*

ENV VCPKG_ROOT=/opt/vcpkg
RUN git clone https://github.com/microsoft/vcpkg.git ${VCPKG_ROOT} \
    && ${VCPKG_ROOT}/bootstrap-vcpkg.sh -disableMetrics
ENV PATH="${VCPKG_ROOT}:${PATH}"

WORKDIR /opt/src
RUN git clone --recursive https://github.com/MrNeRF/LichtFeld-Studio.git . \
    && git checkout ${LFS_REF} \
    && git submodule update --init --recursive

# vcpkg's own binary cache, backed by the GitHub Actions cache service
# (secrets mounted below, exported by the workflow's "Export GitHub
# Actions cache variables" step). This caches per-PACKAGE build output,
# independent of Docker's own layer cache - so if a transient failure
# (like a flaky mirror mid-download) kills the install after 50 of 86
# packages already built successfully, a retry restores those 50 from
# cache instead of recompiling them, and only rebuilds what's left.
RUN --mount=type=secret,id=actions_cache_url \
    --mount=type=secret,id=actions_runtime_token \
    export ACTIONS_CACHE_URL="$(cat /run/secrets/actions_cache_url 2>/dev/null || true)" \
    && export ACTIONS_RUNTIME_TOKEN="$(cat /run/secrets/actions_runtime_token 2>/dev/null || true)" \
    && export VCPKG_BINARY_SOURCES="clear;x-gha,readwrite" \
    && cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake \
        -DBUILD_CUDA_PTX_ONLY=ON \
        -DBUILD_PORTABLE=ON \
        -DBUILD_CUDA_MIN_SM=75 \
        -DCUDA_DEVICE_DEBUG=OFF \
        -DBUILD_TESTS=OFF \
    && cmake --build build -- -j$(nproc) \
    && cmake --install build --prefix /opt/lichtfeld \
    && rm -rf ${VCPKG_ROOT}/buildtrees ${VCPKG_ROOT}/downloads build

# Discover the actual runtime .so closure via ldd instead of guessing which
# apt runtime packages the binary needs - copies every non-system shared
# library the built binary links against into a vendor-libs dir that ships
# in stage 2. (CUDA driver libs are NOT among these - RunPod's NVIDIA
# container runtime mounts those into the container at pod start.)
RUN BIN="$(find /opt/lichtfeld/bin -maxdepth 1 -type f -executable | head -n1)" \
    && echo "$BIN" > /opt/lichtfeld/.entrypoint-bin \
    && mkdir -p /opt/lichtfeld/vendor-libs \
    && ldd "$BIN" | awk '{print $3}' | grep '^/' | sort -u \
        | xargs -I{} sh -c 'cp -L {} /opt/lichtfeld/vendor-libs/ 2>/dev/null || true'

########################################
# Stage 2: runtime
########################################
ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# openssh-server: NOT included by default, and RunPod only auto-configures
# SSH for its own stock templates - a custom image needs this added
# explicitly (see the header comment / oblaQ's prior COLMAP-base image,
# where this was missing on the first deploy attempt).
# xvfb + xauth: safety net in case LichtFeld's --headless flag still tries
# to create a GL context on startup (undocumented either way) - the
# lichtfeld-headless wrapper below runs the binary under xvfb-run
# automatically whenever no DISPLAY is set, so this never needs manual
# attention.
# unzip: confirmed load-bearing - oblaQ/Scripts/session_commands.ps1 unzips
# images.zip/masks.zip/scripts.zip on every fresh pod. zip: not actually
# used pod-side (zipping happens locally on Windows via Compress-Archive
# before upload) but trivial to include for symmetry.
# python3-pip + gdown: session_commands.ps1's real dataset-transfer path is
# `gdown --continue <drive-id>` for the 11GB+ images.zip/masks.zip -
# deliberately preferred over scp'ing from home ("130-200mbps vs volatile
# home upload"). ffmpeg and a COLMAP vocab tree are deliberately NOT
# included - neither shows up anywhere in the current "rig" pipeline path
# (ffmpeg only appears in the superseded video-processing path; the vocab
# tree is a COLMAP loop_detection feature, unrelated to LichtFeld, and is
# noted in session_commands.ps1 as currently disabled/unused anyway).
#
# tmux: load-bearing for anything but the shortest runs - training at
# `-i 30000` can take hours, and without a detachable session the process
# dies the moment your SSH connection drops. Not optional in practice.
# htop, nano, vim, wget: not strictly required, but included because
# LichtFeld's own dev Dockerfile (docker/Dockerfile upstream) ships all
# four - matching that rather than guessing, and they're cheap.
RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server \
        libglu1-mesa libgl1 libegl1 libgtk-3-0 \
        libx11-6 libxrandr2 libxinerama1 libxcursor1 libxi6 \
        libgomp1 libgfortran5 \
        xvfb xauth \
        zip unzip wget \
        tmux htop nano vim \
        python3 python3-pip \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir --break-system-packages gdown

COPY --from=builder /opt/lichtfeld /opt/lichtfeld
ENV LD_LIBRARY_PATH="/opt/lichtfeld/lib:/opt/lichtfeld/vendor-libs:${LD_LIBRARY_PATH}"
ENV PATH="/opt/lichtfeld/bin:/opt/lichtfeld/vendor-libs:${PATH}"

COPY lichtfeld-headless /usr/local/bin/lichtfeld-headless
RUN chmod +x /usr/local/bin/lichtfeld-headless

# Live training monitor: LichtFeld's CLI has an undocumented (not in the
# wiki, confirmed only by reading argument_parser.cpp) TCP signals/events
# feature - `--tcp-connection --tcp-server-port <p> --tcp-broadcast-port
# <p>` - almost certainly what lets a local LichtFeld Studio GUI connect to
# a remote headless training run and watch it live, the same role
# Nerfstudio's ns-viewer plays for ns-train. lichtfeld-headless below
# enables this by default whenever --headless is passed (see that file for
# the opt-out). These two ports need to be reachable from your laptop, so
# EXPOSE here is necessary but not sufficient - RunPod's own pod config
# also needs matching TCP port mappings added (see README's Deploying
# section). Overridable at build time if 8090/8091 collide with something
# else in your setup.
ARG LFS_TCP_SERVER_PORT=8090
ARG LFS_TCP_BROADCAST_PORT=8091
ENV LFS_TCP_SERVER_PORT=${LFS_TCP_SERVER_PORT}
ENV LFS_TCP_BROADCAST_PORT=${LFS_TCP_BROADCAST_PORT}
EXPOSE ${LFS_TCP_SERVER_PORT} ${LFS_TCP_BROADCAST_PORT}

# --- SSH setup (build-time config; runtime key injection happens in start.sh) ---
RUN mkdir -p /var/run/sshd /root/.ssh && chmod 700 /root/.ssh \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config

COPY start.sh /root/start.sh
RUN chmod +x /root/start.sh

WORKDIR /root
ENTRYPOINT ["/root/start.sh"]
