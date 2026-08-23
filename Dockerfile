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

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04 AS test

RUN --mount=type=bind,from=colmap-base,target=/colmap-base \
    COLMAP_BIN="$(find /colmap-base -maxdepth 5 -type f -name colmap -executable 2>/dev/null | head -n1)" \
    && test -n "$COLMAP_BIN" || (echo "ERROR: colmap binary not found in dakord/oblaq-colmap-base:latest" >&2 && exit 1) \
    && mkdir -p /opt/colmap/bin /opt/colmap/vendor-libs /opt/colmap/share \
    && cp -L "$COLMAP_BIN" /opt/colmap/bin/colmap \
    && command -v chroot >/dev/null || (echo "ERROR: chroot not available in the runtime base image" >&2 && exit 1) \
    && COLMAP_REL="${COLMAP_BIN#/colmap-base}" \
    && COLMAP_LIBS="$(chroot /colmap-base ldd "$COLMAP_REL")" \
    && echo "$COLMAP_LIBS" | awk '{print $3}' | grep '^/' \
        | grep -vE '/(libc|libm|libpthread|libdl|librt|libresolv|libnsl|libutil|libcrypt|ld-linux[^/]*|libstdc\+\+|libgcc_s)\.so' \
        | sort -u \
        | xargs -I{} sh -c 'cp -L "/colmap-base{}" /opt/colmap/vendor-libs/ 2>/dev/null || true' \
    && (echo "$COLMAP_LIBS" | grep -i "not found" \
        && echo "WARNING: colmap has unresolved shared library dependencies listed above - it will likely fail to run on the pod" \
        || true) \
    && VOCAB_FILES="$(find /colmap-base -iname '*vocab*tree*' -type f 2>/dev/null)" \
    && if [ -n "$VOCAB_FILES" ]; then \
         echo "$VOCAB_FILES" | xargs -I{} cp -L {} /opt/colmap/share/; \
       else \
         echo "WARNING: no vocab tree file found in dakord/oblaq-colmap-base:latest - vocab-tree matching won't be available, exhaustive/sequential matchers still work fine"; \
       fi

ENV LD_LIBRARY_PATH="/opt/colmap/vendor-libs:${LD_LIBRARY_PATH}"
ENV PATH="/opt/colmap/bin:${PATH}"

# The actual proof: does the copied binary run at all? This is the part a
# "the build succeeded" check does NOT cover - the COLMAP step can copy
# files just fine and still hand you a binary that fails to load at
# runtime, which is exactly what happened twice already. ldd here uses
# THIS stage's own resolver deliberately (no chroot) - that's the real
# production configuration (LD_LIBRARY_PATH + vendor-libs), so this proves
# out the actual runtime path, not just the build-time copy step.
# GPU access isn't available on a GitHub Actions runner (or most local
# Docker setups), so `colmap -h` is as far as this can verify - it doesn't
# touch CUDA, but it does exercise every CPU-side dynamic library colmap
# needs, which is the actual thing that broke twice.
RUN echo "=== ldd on the assembled binary (production LD_LIBRARY_PATH) ===" \
    && ldd /opt/colmap/bin/colmap \
    && echo "" \
    && echo "=== checking for any unresolved dependencies ===" \
    && (ldd /opt/colmap/bin/colmap | grep -i "not found" \
        && (echo "FAIL: unresolved dependencies listed above" >&2 && exit 1) \
        || echo "OK: nothing unresolved") \
    && echo "" \
    && echo "=== colmap -h ===" \
    && colmap -h \
    && echo "" \
    && echo "=== vendor-libs contents ===" \
    && ls -la /opt/colmap/vendor-libs \
    && echo "" \
    && echo "=== vocab tree ===" \
    && ls -la /opt/colmap/share 2>&1 || echo "(none found)"
