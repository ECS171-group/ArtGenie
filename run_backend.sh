#!/bin/bash

# Store the original requirements.txt
cp src/requirements.txt src/requirements.txt.bak

# Remove tensorflow from requirements.txt temporarily
sed -i '/tensorflow/d' src/requirements.txt

# Run the Docker command
docker run -e HSA_OVERRIDE_GFX_VERSION=10.3.0 -it --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v ./src:/app rocm/tensorflow:latest

# Restore the original requirements.txt
mv src/requirements.txt.bak src/requirements.txt

