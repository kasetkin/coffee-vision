#!/bin/bash
# Makes /dev/dri/renderD* (the AMD iGPU's compute node, passed in via
# --device=/dev/dri in cpu/devcontainer.json) usable by the non-root
# container user. Runs on every container start (postStartCommand), because
# the device's group ownership is a host property we can't know at image
# build time — it can differ machine to machine.
#
# Why this lives in the CPU Dockerfile, of all places: "cpu" here describes
# which PyTorch backend that image uses (no NVIDIA GPU, so no CUDA) — it
# doesn't mean the machine has no GPU-like hardware at all. It still has the
# AMD integrated GPU, and that's exactly what this script is for: letting
# OpenCV's OpenCL path (unrelated to PyTorch/CUDA) reach it. Dockerfile.gpu
# is for a cloud NVIDIA box, which has no AMD graphics stack and thus no
# /dev/dri render node here to manage — that's why it doesn't ship this file
# at all, not because GPU-related setup universally belongs there instead.
set -euo pipefail
shopt -s nullglob

devices=(/dev/dri/renderD*)
if [ ${#devices[@]} -eq 0 ]; then
    echo "No /dev/dri/renderD* device present — skipping OpenCL device setup."
    exit 0
fi

for dev in "${devices[@]}"; do
    gid=$(stat -c '%g' "$dev")
    group_name=$(getent group "$gid" | cut -d: -f1 || true)
    if [ -z "$group_name" ]; then
        group_name="hostgpu$gid"
        groupadd -g "$gid" "$group_name"
    fi
    usermod -aG "$group_name" vscode
    echo "Added vscode to group $group_name (gid $gid) for $dev"
done
