# syntax=docker/dockerfile:1
#
# Standalone test for JUST the COLMAP-bundling step from the main Dockerfile.
# Skips LichtFeld's ~60-90 min GCC-14/CUDA/vcpkg compile entirely - this
# builds in a couple minutes since it only pulls two base images and runs
# the one RUN step being tested. Use this to validate any change to that
# step before spending two hours on a full rebuild to find out it broke.
#
# Build locally (if you have Docker Desktop):
#   docker build -f Dockerfile.colmap-test -t colmap-test .
# Or via the test-colmap.yml workflow on GitHub Actions if you don't.
#
# The logic below is copy-pasted VERBATIM from the main Dockerfile's COLMAP
# step - if you change one, change the other, or better, diff them before
# trusting a "passed" result here.

ARG CUDA_VERSION=12.8.0

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
# ldd's own /bin/bash libc.so.6 resolution" GLIBC-mismatch bug (see the
# comment on the chroot call below), because every path in it lives
# inside this SAME chroot'd filesystem.
FROM colmap-base AS colmap-base-env
RUN echo "$LD_LIBRARY_PATH" > /captured_ld_library_path.txt

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04 AS test

# The ,rw here is required: a bind mount is read-only by default (per
# BuildKit's own docs), and chroot'ing into it means glibc's own `ldd`
# script - which does `> /dev/null` internally - fails outright with
# "/dev/null: Read-only file system" and then misreports the target as
# "not a dynamic executable". rw only affects this RUN instruction (nothing
# written here is committed to any layer), and we're not relying on
# anything actually being written back into colmap-base anyway - this is
# purely to let ldd's own internal housekeeping succeed.
RUN --mount=type=bind,from=colmap-base,target=/colmap-base,rw \
    --mount=type=bind,from=colmap-base-env,source=/captured_ld_library_path.txt,target=/captured_ld_library_path.txt \
    COLMAP_BIN="$(find /colmap-base -maxdepth 5 -type f -name colmap -executable 2>/dev/null | head -n1)" \
    && test -n "$COLMAP_BIN" || (echo "ERROR: colmap binary not found in dakord/oblaq-colmap-base:latest" >&2 && exit 1) \
    && mkdir -p /opt/colmap/bin /opt/colmap/vendor-libs /opt/colmap/share \
    && cp -L "$COLMAP_BIN" /opt/colmap/bin/colmap \
    && command -v chroot >/dev/null || (echo "ERROR: chroot not available in the runtime base image" >&2 && exit 1) \
    && COLMAP_REL="${COLMAP_BIN#/colmap-base}" \
    && COLMAP_BASE_LDPATH="$(cat /captured_ld_library_path.txt)" \
    && COLMAP_LIBS="$(chroot /colmap-base /bin/sh -c "LD_LIBRARY_PATH='$COLMAP_BASE_LDPATH' ldd '$COLMAP_REL'")" \
    && echo "$COLMAP_LIBS" | awk '{print $3}' | grep '^/' \
        | grep -vE '/(libc|libm|libpthread|libdl|librt|libresolv|libnsl|libutil|libcrypt|ld-linux[^/]*|libstdc\+\+|libgcc_s)\.so' \
        | sort -u \
        | xargs -I{} sh -c 'cp -L "/colmap-base{}" /opt/colmap/vendor-libs/ 2>/dev/null || true' \
    # Kept in sync with the main Dockerfile's hardened version of this
    # check. Rewritten from the earlier `(X && (echo E; exit 1) || true)`
    # form to a plain if/exit: that form ALWAYS exited 0 regardless of
    # whether "not found" matched - `X && Y` counts as failed whenever Y's
    # `exit 1` runs, which then triggers the trailing `|| true`, silently
    # discarding the failure. Confirmed for real: the CI run this was
    # meant to gate printed the ERROR line and then continued to a
    # reported build success anyway. `exit 1` inside an `if` body ends the
    # whole RUN's shell script immediately, so it can't be swallowed by
    # anything later in the chain.
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

# Matches the main Dockerfile: no global PATH/LD_LIBRARY_PATH ENV for
# colmap (breaks SSH-session PATH persistence and shadows the system
# liblzma.so.5 for every process via colmap-base's older vendored copy,
# see the main Dockerfile's comment on this same step) - a
# /usr/local/bin/colmap wrapper instead. Keep this in sync with the main
# Dockerfile's version, per the header note above.
RUN printf '#!/bin/sh\nexec env LD_LIBRARY_PATH="/opt/colmap/vendor-libs:${LD_LIBRARY_PATH}" /opt/colmap/bin/colmap "$@"\n' \
        > /usr/local/bin/colmap \
    && chmod +x /usr/local/bin/colmap

# The actual proof: does the copied binary run at all? This is the part a
# "the build succeeded" check does NOT cover - the COLMAP step can copy
# files just fine and still hand you a binary that fails to load at
# runtime, which is exactly what happened twice already. ldd here scopes
# LD_LIBRARY_PATH to just this one invocation - that's the real production
# configuration (see the wrapper above), so this proves out the actual
# runtime path, not just the build-time copy step.
# GPU access isn't available on a GitHub Actions runner (or most local
# Docker setups), so `colmap -h` is as far as this can verify - it doesn't
# touch CUDA, but it does exercise every CPU-side dynamic library colmap
# needs, which is the actual thing that broke twice. Note this still won't
# catch a dlopen'd-only dependency like libGLEW.so.2.2 (see the main
# Dockerfile's runtime apt-get list) - ldd, and by extension this whole
# smoke test, only sees libraries linked directly at build time.
# Rewritten from the original `(A && B && ... && LAST || echo "(none
# found)")` form, which had TWO instances of the exact masking bug fixed
# above in the vendoring step: (1) the inner "not found" check used the
# same `X && (echo FAIL; exit1) || echo OK` pattern that always resolved
# to "OK" regardless of X, and (2) far worse, the trailing `|| echo "(none
# found)"` was meant only to tolerate a missing vocab-tree `ls`, but since
# `&&` short-circuits, ANY earlier command failing in this chain -
# including colmap -h actually crashing on a missing shared library, which
# is the one thing this whole test exists to catch - also fell through to
# that same fallback and reported success. Confirmed for real: the CI run
# this was meant to gate showed `error while loading shared libraries:
# libcudart.so.13: cannot open shared object file`, immediately followed
# by "(none found)" and a clean `DONE 0.1s`, with no `##[error]` anywhere
# in the log. Split into two RUN steps below so the vocab-tree fallback is
# structurally isolated and can never again absorb an unrelated failure.
RUN echo "=== ldd on the assembled binary (production LD_LIBRARY_PATH) ===" \
    && LD_LIBRARY_PATH="/opt/colmap/vendor-libs:${LD_LIBRARY_PATH}" ldd /opt/colmap/bin/colmap \
    && echo "" \
    && echo "=== checking for any unresolved dependencies ===" \
    && if LD_LIBRARY_PATH="/opt/colmap/vendor-libs:${LD_LIBRARY_PATH}" ldd /opt/colmap/bin/colmap | grep -qi "not found"; then \
         echo "FAIL: unresolved dependencies listed above" >&2; \
         exit 1; \
       else \
         echo "OK: nothing unresolved"; \
       fi \
    && echo "" \
    && echo "=== colmap -h (via the /usr/local/bin wrapper, same path production uses) ===" \
    && colmap -h \
    && echo "" \
    && echo "=== vendor-libs contents ===" \
    && ls -la /opt/colmap/vendor-libs

# Isolated in its own RUN step so a missing vocab tree (tolerated - see
# the WARNING above) can never mask a failure from the step before it.
RUN echo "=== vocab tree ===" \
    && (ls -la /opt/colmap/share 2>&1 || echo "(none found)")
