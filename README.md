# LFStudio — LichtFeld Studio, baked into a reusable RunPod image

Builds [MrNeRF/LichtFeld-Studio](https://github.com/MrNeRF/LichtFeld-Studio) once, headless, and pushes the result to
Docker Hub as `dakord/lichtfeld-studio`, so future RunPod pods boot straight into a working binary instead of paying
the ~60-90 min compile every time.

## How it fits together

- **Dockerfile** — two stages. Stage 1 (`builder`) compiles LichtFeld against CUDA 12.8 (GCC-14, vcpkg, portable
  PTX targeting SM 7.5+, so the one image works on whatever GPU model RunPod assigns you). Stage 2 (`runtime`) is
  the slim image RunPod actually pulls: just the compiled binary + the runtime libraries it actually needs
  (discovered via `ldd` at build time, not guessed), plus SSH so you can connect the same way you already do on
  other oblaQ pods.
- **start.sh** — runs at container *start*, not build time. Writes RunPod's `$PUBLIC_KEY` into `authorized_keys`
  and starts `sshd`. This is the same pattern already proven in `oblaQ/Git/files/start.sh` for the COLMAP/Nerfstudio
  image — reused here rather than re-debugging the same "Connection refused" issue that pattern was built to fix.
- **lichtfeld-headless** — a wrapper on `PATH` inside the container. Run it manually once you've SSH'd in, same as
  you already do for `ns-train` etc. — ideally inside `tmux` (`tmux new -s train`, then run the command below, then
  `Ctrl-b d` to detach) since a multi-hour training run otherwise dies the moment your SSH connection drops:
  ```bash
  lichtfeld-headless --headless --train -d /workspace/data -o /workspace/output -i 30000 -r 1
  ```
  It wraps the binary in `xvfb-run` automatically if no `DISPLAY` is set, as a safety net in case `--headless`
  still wants a GL context internally (undocumented either way — costs nothing to have, avoids a surprise crash
  if it turns out to matter).

  It also auto-enables LichtFeld's TCP signals/events feature whenever `--headless` is passed — this is
  undocumented in the project's wiki, confirmed only by reading `argument_parser.cpp` directly, but it's almost
  certainly what lets a local LichtFeld Studio GUI connect to a remote headless run and watch training live, the
  same role Nerfstudio's `ns-viewer` plays for `ns-train`. Ports default to `8090` (server) / `8091` (broadcast),
  baked in at image build time — see **Live training monitor** below for the RunPod-side setup this needs. Skip it
  for a given run with `LFS_NO_TCP=1 lichtfeld-headless ...`, or pass `--tcp-connection` yourself if you want
  different ports for just that run.

## One-time setup

1. In this repo's **Settings → Secrets and variables → Actions**, add two repository secrets:
   - `DOCKERHUB_USERNAME` = `dakord`
   - `DOCKERHUB_TOKEN` = a Docker Hub **access token** (not your password) — create one at
     [hub.docker.com → Account Settings → Security → New Access Token](https://hub.docker.com/settings/security),
     with Read/Write scope.
2. Commit these files (`Dockerfile`, `start.sh`, `lichtfeld-headless`, `.github/workflows/build-image.yml`) to the
   repo.

## Building the image

Go to **Actions → Build and push LichtFeld Studio image → Run workflow**. Leave `lfs_ref` as `master` to track
upstream's latest, or pin a specific tag/commit for reproducibility.

This runs on GitHub's free runners (no GPU needed to compile — `nvcc` cross-compiles PTX without a physical GPU
present). Expect the *first* run to take a while — GitHub's standard runners are only 2-core, so this build won't
necessarily be faster than doing it once on a RunPod pod, just free. Later runs reuse unchanged Docker layers via
GitHub Actions' build cache, so rebuilding after bumping `lfs_ref` should be quicker than the first build.

When it finishes, the image is at `dakord/lichtfeld-studio:latest` on Docker Hub (public).

## Deploying on RunPod

1. New Pod → **Custom Container** (not a template).
2. Container Image: `dakord/lichtfeld-studio:latest`.
3. Leave the default container start command alone — `start.sh` is the image's `ENTRYPOINT` and handles SSH +
   keeping the pod alive on its own, the same way your other custom oblaQ images do.
4. Attach whatever volume you're using for data (network volume, or just the pod's container disk if this is a
   short-lived run) and mount it somewhere under `/workspace`.
5. **If you want live training monitoring from your local LichtFeld Studio GUI** (see below), add TCP port mappings
   for `8090` and `8091` in the pod's network config *before* deploying — RunPod's pod editor has an "Expose TCP
   Ports" field, add both there. If you skip this, training still works fine — you just can't watch it live and
   would check progress via `--log-file` output or by SSH'ing back in periodically instead.
6. Once the pod is running, SSH in exactly as usual, then run training inside `tmux` so it survives a dropped
   connection:
   ```bash
   tmux new -s train
   lichtfeld-headless --headless --train -d /workspace/data -o /workspace/output -i 30000 -r 1
   # Ctrl-b d to detach; `tmux attach -t train` to come back later
   ```

No 60-90 min wait — that already happened once, in GitHub Actions, for free.

## Live training monitor

`lichtfeld-headless` passes `--tcp-connection --tcp-server-port 8090 --tcp-broadcast-port 8091` automatically
whenever you pass `--headless` (see above to disable or override). Once RunPod's TCP ports are mapped (step 5
above), RunPod will show you the external host:port pair it mapped `8090`/`8091` to — connect your local
LichtFeld Studio GUI to that address to watch training progress live, the same way you'd watch an `ns-train` run
via `ns-viewer`. This whole feature is undocumented upstream (found by reading the CLI source, not the wiki), so
treat the exact connect-from-GUI steps as something to confirm on your first real run rather than as guaranteed.

## Notes / open questions worth checking on first real run

- The actual executable name on Linux comes from `install(TARGETS ${PROJECT_NAME} ...)`, i.e. literally
  `LichtFeld-Studio` (matching the CMake project name/casing) — `lichtfeld-headless` looks this up dynamically
  rather than hardcoding it, so a casing mismatch with the wiki's lowercase CLI examples shouldn't matter, but
  worth a quick sanity check (`lichtfeld-headless --help`) after your first pod boot.
- Whether `--headless` genuinely avoids needing a GL context isn't documented either way upstream — the
  `xvfb-run` fallback in `lichtfeld-headless` covers this either way, but if the first run shows a GL/EGL error
  despite that, that's the first place to look.
- `master` is a moving target — pin `lfs_ref` to a tag once you've confirmed a build works well for your dataset,
  so a future rebuild doesn't silently change behavior on you.
