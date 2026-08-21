#!/bin/bash
# Runs at container START (not build time) - sets up SSH access using
# RunPod's $PUBLIC_KEY environment variable convention, then starts sshd.
#
# Adapted verbatim from oblaQ/Git/files/start.sh (the COLMAP/Nerfstudio
# pod image) - same reasoning applies here: $PUBLIC_KEY is only set when
# the pod actually launches, so this has to run at container start rather
# than as a build-time RUN step.

# Generate host keys if they don't exist yet (first boot of this container)
if [ ! -f /etc/ssh/ssh_host_rsa_key ]; then
    ssh-keygen -A
fi

# Write the pod's authorized public key, if RunPod provided one
if [ -n "$PUBLIC_KEY" ]; then
    echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi

# Start sshd in the background
service ssh start

# Keep the container running (sshd alone would let the container exit
# immediately otherwise, since it's not the foreground process)
tail -f /dev/null
