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
# nasm/autoconf/autoconf-archive/automake/libtool: needed by vcpkg ports
# further down the dependency tree (x264's asm routines need nasm on PATH
# directly - vcpkg_find_acquire_program(NASM) doesn't build its own; other
# autotools-based ports expect these on the system rather than bundling
# their own, unlike vcpkg-make which does fetch its own automake).
# libcudnn9-cuda-12: ONNX Runtime's CUDA execution provider depends on
# cuDNN 9 (per LFS's own CMakeLists.txt, ~line 454) - the CUDA *toolkit*
# devel image does NOT bundle cuDNN (that's a separate NVIDIA library), so
# without this, `cmake --install`'s runtime-dependency resolution fails
# outright with "Could not resolve runtime dependencies: libcudnn.so.9".
# The NVIDIA apt repo this comes from is already registered in this base
# image (baked in via cuda-keyring when nvidia/cuda itself was built), so
# no extra apt source/key setup is needed here - just the install.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates gnupg wget curl git zip unzip tar pkg-config \
        build-essential \
        gcc-14 g++-14 gfortran-14 \
        ninja-build python3 python3-pip \
        nasm autoconf autoconf-archive automake libtool \
        libcudnn9-cuda-12 \
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

# LichtFeld's own build hardcodes -march=native for Release builds
# (src/core/CMakeLists.txt, confirmed by reading it directly) - no CMake
# option exposes it, so it has to be patched here. -march=native bakes in
# whatever exact CPU instruction set THIS GitHub Actions runner happens to
# have. This image then runs on a totally different RunPod host CPU, which
# can lack one of those instructions - confirmed on a real deploy as an
# instant "Illegal instruction (core dumped)" the moment the binary starts,
# nothing to do with CUDA/cuDNN. Patched to x86-64-v3 (AVX2 + FMA + BMI2 +
# LZCNT) instead: a portable baseline every modern cloud/server CPU
# supports, and consistent with the -mavx2 -mfma the project already opts
# into explicitly a few lines below the -march=native line in that same file.
RUN sed -i 's/-march=native/-march=x86-64-v3/' src/core/CMakeLists.txt

# Build flags:
#   BUILD_CUDA_PTX_ONLY + BUILD_PORTABLE: JIT PTX at first run instead of
#     baking one fixed SM target, so this one image works whatever GPU
#     model RunPod happens to hand you (chosen over pinning to a single
#     GPU's compute capability).
#   BUILD_CUDA_MIN_SM=75: matches LichtFeld's own default minimum (SM 7.5),
#     set explicitly here for clarity rather than relying on the default.
#   CUDA_DEVICE_DEBUG=OFF: CMakeLists.txt defaults this ON, which ships a
#     -G debug-instrumented (much slower) CUDA binary - turned off since
#     this image is for actual training runs, not debugging LichtFeld itself.
#   BUILD_PYTHON_STUBS=OFF: defaults ON upstream (src/python/CMakeLists.txt)
#     and is wired as an ALL target, so it normally blocks the whole build.
#     Generating the stubs requires actually importing the freshly-built
#     `lichtfeld` Python module, which dlopens libcuda.so.1 - the real NVIDIA
#     driver's library, not something the CUDA *toolkit* ships. GitHub's
#     runners have no GPU/driver, so that import always fails here. This
#     only skips optional IDE autocomplete stub files, not the actual binary
#     or its CUDA functionality at runtime on RunPod (where a real driver is
#     present).
#
# The two --mount=type=cache mounts give vcpkg's own binary/download cache a
# home that BuildKit can persist across builds (the workflow's
# buildkit-cache-dance step saves/restores these via actions/cache - a plain
# --mount=type=cache alone resets on every GitHub Actions run, since each job
# gets a fresh builder). Without this, all ~89 vcpkg packages - including
# USD, the single slowest one - rebuild from scratch on every image rebuild.
# (An earlier attempt used vcpkg's `x-gha` binary-caching backend directly -
# that backend was silently removed from vcpkg and only printed a warning,
# so it never actually cached anything; this replaces it with vcpkg's
# `files` backend pointed at these cache-mount directories instead.)
RUN --mount=type=cache,target=/vcpkg-binary-cache \
    --mount=type=cache,target=/vcpkg-downloads-cache \
    export VCPKG_BINARY_SOURCES="clear;files,/vcpkg-binary-cache,readwrite" \
    && export VCPKG_DOWNLOADS=/vcpkg-downloads-cache \
    && cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake \
        -DBUILD_CUDA_PTX_ONLY=ON \
        -DBUILD_PORTABLE=ON \
        -DBUILD_CUDA_MIN_SM=75 \
        -DCUDA_DEVICE_DEBUG=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_PYTHON_STUBS=OFF \
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
# Stage: COLMAP (reused from oblaQ's existing CUDA-enabled build)
########################################
# No RUN/COPY needed here - this stage exists purely so the runtime stage
# below can --mount=type=bind into its filesystem to pull out the colmap
# binary, its vocab tree, and its runtime .so closure, without copying this
# whole ~660MB image into a layer.
FROM dakord/oblaq-colmap-base:latest AS colmap-base

# `chroot` below only gets colmap-base's filesystem, not its Docker image
# config (ENV/PATH/etc. are metadata, not files) - if colmap-base, like
# nvidia/cuda base images generally, exposes some of its CUDA/math
# libraries via an ENV-set LD_LIBRARY_PATH rather than (or in addition to)
# ldconfig-registered paths, a bare chroot ldd would wrongly report those
# "not found" even though colmap-base's own build step (`RUN colmap -h |
# grep -q "with CUDA"`, which passed) and any real `docker run` of it
# resolve them fine. This stage DOES see colmap-base's actual image env,
# so its LD_LIBRARY_PATH gets captured to a file and fed into the chroot
# ldd call below - closing that gap instead of guessing at the path.
# This is deliberately colmap-base's OWN captured value, not some
# externally-guessed path - it doesn't reintroduce the earlier "poisoned
# ldd's own /bin/bash libc.so.6 resolution" GLIBC-mismatch bug described
# in the chroot call's own comment below, because every path in it lives
# inside this SAME chroot'd filesystem.
FROM colmap-base AS colmap-base-env
RUN echo "$LD_LIBRARY_PATH" > /captured_ld_library_path.txt

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
# libglew2.2: colmap's feature_extractor dlopens GLEW for its SiftGPU path
# rather than linking it directly - confirmed by hitting "error while
# loading shared libraries: libGLEW.so.2.2" on a real pod even though the
# ldd-driven vendoring above ran clean (ldd only sees direct link
# dependencies, not dlopen'd ones, so it can't catch this class of
# missing lib). Installed explicitly here rather than relying on
# vendoring to happen to catch it.
# libcudnn9-cuda-12: ONNX Runtime's CUDA execution provider needs this at
# actual runtime, not just at build time. LFS's own CMakeLists.txt portable-
# bundling logic deliberately EXCLUDES anything under /usr/lib* from being
# copied into the install tree (it assumes standard system libs are already
# present on the target machine) - true for glibc/libstdc++, not true for
# cuDNN, which apt installs into exactly that excluded path. So installing
# this only in the builder stage would satisfy cmake --install's dependency
# check but silently ship a pod that's missing cuDNN at actual train time -
# it has to be installed here too, in the image that's actually deployed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server \
        libglu1-mesa libgl1 libegl1 libgtk-3-0 \
        libx11-6 libxrandr2 libxinerama1 libxcursor1 libxi6 \
        libgomp1 libgfortran5 \
        libcudnn9-cuda-12 \
        libglew2.2 \
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

# COLMAP: reused from oblaQ's already-built, already-proven CUDA-enabled
# COLMAP 4.1.1 image (dakord/oblaq-colmap-base) instead of compiling it from
# source here - a from-source COLMAP build needs Ceres, Qt6, CGAL, boost,
# and OpenImageIO on top of the CUDA toolkit, a much bigger and riskier
# addition than reusing a binary that's already running COLMAP jobs in
# production for the oblaQ pipeline. That image's own Dockerfile lives in a
# separate, private repo this one can't read directly, so instead of
# assuming a fixed install path, this discovers the binary/vocab-tree/lib
# closure by searching the actual image at build time - the same ldd-driven
# technique already used above for the LichtFeld binary itself.
# Caveat inherited, not introduced, by reuse: if oblaq-colmap-base's CUDA
# feature extraction was built for specific GPU compute capabilities rather
# than LichtFeld's portable-PTX approach, an unusual RunPod GPU could
# theoretically hit a mismatch - not something this Dockerfile can detect,
# though the oblaQ pipeline already runs this same image across RunPod's
# GPU fleet in production, which is real-world evidence it's fine in
# practice.
# The ,rw here is required: a bind mount is read-only by default (per
# BuildKit's own docs), and chroot'ing into it means glibc's own `ldd`
# script - which does `> /dev/null` internally - fails outright with
# "/dev/null: Read-only file system" and then misreports the target as
# "not a dynamic executable" (caught via the fast standalone
# Dockerfile.colmap-test, not this 60-90 min build - that's exactly why
# that test file exists). rw only affects this RUN instruction (nothing
# written here is committed to any layer); nothing actually needs to be
# written back into colmap-base, this is purely so ldd's own internal
# housekeeping succeeds.
RUN --mount=type=bind,from=colmap-base,target=/colmap-base,rw \
    --mount=type=bind,from=colmap-base-env,source=/captured_ld_library_path.txt,target=/captured_ld_library_path.txt \
    COLMAP_BIN="$(find /colmap-base -maxdepth 5 -type f -name colmap -executable 2>/dev/null | head -n1)" \
    && test -n "$COLMAP_BIN" || (echo "ERROR: colmap binary not found in dakord/oblaq-colmap-base:latest" >&2 && exit 1) \
    && mkdir -p /opt/colmap/bin /opt/colmap/vendor-libs /opt/colmap/share \
    && cp -L "$COLMAP_BIN" /opt/colmap/bin/colmap \
    # Two earlier attempts here both broke on the same root cause: ldd needs
    # to resolve against colmap-base's OWN glibc/ld.so.cache (which already
    # has cuDSS's non-standard path registered via its own Dockerfile's
    # `ldconfig` run), not this stage's. Pointing LD_LIBRARY_PATH at
    # colmap-base's lib dirs looked like the fix, but `ldd` is itself a bash
    # script - even scoping the env var to just that one command still
    # poisons the /bin/bash process ldd execs internally, which then tries
    # to resolve ITS OWN libc.so.6 through that same search path and picks
    # up colmap-base's older (Ubuntu 22.04/glibc 2.35) one instead of this
    # stage's - confirmed directly from two separate failed build logs,
    # both crashing on "GLIBC_2.38 not found (required by /bin/bash)".
    # chroot sidesteps this entirely: colmap-base's own ldd, bash, and libc
    # all come from the SAME consistent filesystem, so nothing gets mixed,
    # and its own ld.so.cache is used automatically - no LD_LIBRARY_PATH
    # needed at all anymore. ldd's output paths are then relative to that
    # chroot, so they get re-prefixed with /colmap-base to `cp` them out.
    && command -v chroot >/dev/null || (echo "ERROR: chroot not available in the runtime base image" >&2 && exit 1) \
    && COLMAP_REL="${COLMAP_BIN#/colmap-base}" \
    && COLMAP_BASE_LDPATH="$(cat /captured_ld_library_path.txt)" \
    && COLMAP_LIBS="$(chroot /colmap-base /bin/sh -c "LD_LIBRARY_PATH='$COLMAP_BASE_LDPATH' ldd '$COLMAP_REL'")" \
    # Exclude glibc's own core libs (and the C++/GCC runtime) from what gets
    # vendored - colmap-base is Ubuntu 22.04/glibc 2.35, this runtime stage
    # is Ubuntu 24.04/glibc 2.39, and ldd will list colmap's libc.so.6 as a
    # dependency same as any dynamically-linked binary. Copying THAT older
    # libc into /opt/colmap/vendor-libs and putting it on LD_LIBRARY_PATH
    # would shadow the correct newer glibc for every OTHER process on the
    # pod (not just colmap), reproducing the exact same GLIBC-mismatch crash
    # this whole step already broke on twice - just silently, at pod runtime
    # instead of build time. Standard practice for any cross-distro binary
    # vendoring (AppImage/linuxdeploy do the same) - only colmap's genuine
    # third-party deps (Ceres, cuDSS, Boost, Qt5, OpenCV, etc.) get copied;
    # glibc/libstdc++/libgcc_s always come from the destination image itself.
    && echo "$COLMAP_LIBS" | awk '{print $3}' | grep '^/' \
        | grep -vE '/(libc|libm|libpthread|libdl|librt|libresolv|libnsl|libutil|libcrypt|ld-linux[^/]*|libstdc\+\+|libgcc_s)\.so' \
        | sort -u \
        | xargs -I{} sh -c 'cp -L "/colmap-base{}" /opt/colmap/vendor-libs/ 2>/dev/null || true' \
    # Was warning-only; upgraded to a hard failure once colmap-base v5's
    # jump to CUDA 13.0/Ubuntu 22.04 (from whatever this was last built
    # against) made an actual mismatch here plausible rather than
    # hypothetical - colmap-base's own header comment flags dependent
    # images as needing "matching updates... not done" yet. A silent
    # warning here means a broken image still reports build success;
    # this way a bad dependency closure fails the build instead of
    # surfacing on a paid GPU pod. This checks resolution INSIDE
    # colmap-base's own chroot (i.e. is colmap-base itself broken) -
    # see the second check below, after the wrapper, for whether the
    # ASSEMBLED vendor-libs resolves in THIS (destination) image.
    # Rewritten from the earlier `(X && (echo E; exit 1) || true)` form:
    # that form ALWAYS exited 0 regardless of whether "not found" matched -
    # `X && Y` counts as failed whenever Y's `exit 1` runs, which then
    # triggers the trailing `|| true`, silently discarding the failure.
    # Confirmed for real via Dockerfile.colmap-test's own CI run (identical
    # pattern there): it printed the ERROR line and then reported build
    # success anyway. `exit 1` inside an `if` body ends the whole RUN's
    # shell script immediately, so nothing later in the chain can catch it.
    && if echo "$COLMAP_LIBS" | grep -qi "not found"; then \
         echo "ERROR: colmap has unresolved shared library dependencies even inside colmap-base's own chroot - see above" >&2; \
         exit 1; \
       fi \
    && VOCAB_FILES="$(find /colmap-base -iname '*vocab*tree*' -type f 2>/dev/null)" \
    && if [ -n "$VOCAB_FILES" ]; then \
         echo "$VOCAB_FILES" | xargs -I{} cp -L {} /opt/colmap/share/; \
       else \
         echo "WARNING: no vocab tree file found in dakord/oblaq-colmap-base:latest - vocab-tree matching won't be available, exhaustive/sequential matchers still work fine"; \
       fi
# Deliberately NOT putting /opt/colmap/bin on PATH or vendor-libs on
# LD_LIBRARY_PATH globally, unlike the LichtFeld binary above - two real
# problems hit on an actual running pod:
#   1. PATH doesn't survive reliably: this same ENV PATH addition was set
#      at image build time, yet a freshly-opened SSH session on a live pod
#      still reported "colmap: command not found" - PAM/profile handling
#      on login can reset PATH per-session regardless of what the
#      container's default env declares.
#   2. LD_LIBRARY_PATH pollutes the whole container: colmap-base is Ubuntu
#      22.04 and its vendored libs include an older liblzma.so.5 than this
#      (Ubuntu 24.04) runtime stage ships. With vendor-libs on
#      LD_LIBRARY_PATH globally, that older liblzma shadows the system one
#      for EVERY process, not just colmap - confirmed directly: apt-get
#      install broke mid-session with "version 'XZ_5.4' not found" because
#      dpkg-deb picked up the shadowed lib.
# A thin wrapper at /usr/local/bin/colmap avoids both: /usr/local/bin is
# already on the default PATH for every login mechanism (nothing extra to
# set), and LD_LIBRARY_PATH is scoped to just this one exec instead of the
# whole shell environment.
RUN printf '#!/bin/sh\nexec env LD_LIBRARY_PATH="/opt/colmap/vendor-libs:${LD_LIBRARY_PATH}" /opt/colmap/bin/colmap "$@"\n' \
        > /usr/local/bin/colmap \
    && chmod +x /usr/local/bin/colmap

# Second, complementary check to the one in the vendoring step above: that
# one verifies colmap-base's OWN chroot can resolve every dependency (is
# colmap-base itself broken); this one verifies the ASSEMBLED
# /opt/colmap/vendor-libs actually resolves everything in THIS
# (destination) image, which is the check that actually determines
# whether colmap runs on the pod - colmap-base is Ubuntu 22.04, this
# stage is 24.04, so a clean chroot resolution does not guarantee a clean
# destination resolution. Mirrors Dockerfile.colmap-test's own smoke
# test, run for real here so a broken dependency closure fails THIS
# build too, not just the separate fast test someone has to remember to
# run first. Doesn't touch CUDA (no GPU on a GitHub Actions runner), but
# exercises every CPU-side dynamic library colmap needs - won't catch a
# dlopen'd-only dependency like libGLEW.so.2.2 (ldd only sees direct
# link-time deps), so that class of bug still needs a real pod test.
# Rewritten from `(X && (echo E; exit1) || echo OK)`: same masking bug as
# the vendoring-step check above - X && Y counts as failed whenever Y's
# exit 1 runs, so the trailing `|| echo OK` fired unconditionally and this
# check could never actually fail the build no matter what ldd reported.
RUN if LD_LIBRARY_PATH="/opt/colmap/vendor-libs:${LD_LIBRARY_PATH}" ldd /opt/colmap/bin/colmap | grep -qi "not found"; then \
      echo "ERROR: colmap has unresolved shared library dependencies in the assembled destination image - see above" >&2; \
      exit 1; \
    else \
      echo "OK: colmap's dependency closure resolves cleanly in the destination image"; \
    fi \
    && colmap -h >/dev/null \
    && echo "OK: colmap -h runs via the /usr/local/bin wrapper"

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
# X11Forwarding: pinned explicitly rather than trusting this base image's
# default - needed to view a GUI app (colmap gui, or the LichtFeld GUI
# binary itself) by tunneling it through `ssh -X` to a local X server on
# your machine (VcXsrv/Xming/MobaXterm), instead of standing up a
# VNC/noVNC stack. Rendering happens client-side, so this needs no extra
# EXPOSE/port-mapping - it rides the same SSH port/session already in use.
# xauth (already installed above) is the other half this needs, for the
# per-session MIT-MAGIC-COOKIE auth forwarding relies on.
# Caveat: colmap's GUI links Qt5 (already vendored - see the COLMAP step's
# comment), but Qt loads its X11 "platform plugin" (libqxcb.so) via
# dlopen at runtime, not a normal link - the same class of blind spot
# that made ldd-driven vendoring miss libGLEW.so.2.2 for feature_extractor.
# Untested whether that plugin and its own deps are actually present;
# worth an actual `colmap gui` try over ssh -X before relying on this.
RUN mkdir -p /var/run/sshd /root/.ssh && chmod 700 /root/.ssh \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config \
    && sed -i 's/X11Forwarding no/X11Forwarding yes/' /etc/ssh/sshd_config \
    && (grep -q '^X11Forwarding yes' /etc/ssh/sshd_config || echo 'X11Forwarding yes' >> /etc/ssh/sshd_config)

COPY start.sh /root/start.sh
RUN chmod +x /root/start.sh

WORKDIR /root
ENTRYPOINT ["/root/start.sh"]
