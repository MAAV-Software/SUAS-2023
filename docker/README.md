# Docker

This folder contains the Dockerfiles for building and running the SUAS software in a Docker container.

## Files for development

General idea is that you use the `Dockerfile.dev` to define a development environment that contains all the dependencies like ROS, Mavlink, and PX4 that are needed to build and develop software. You then build and push the image using the `push_dev.sh` script to dockerhub so the next time anyone wants to use it, they dont have to waste time building the whole thing again. Once a certain version or "tag" of the image is on dockerhub, the development environment defined in the `.devcontainer` folder will use the`.devcontainer/Dockerfile` to set up your VSCode with everything you need using the pre-built image from dockerhub for your machine's specific architecture.

**NOTE**: If you want to add more software dependencies and dont manage to build and push the image to dockerhub, you can still just add them to `.devcontainer/Dockerfile` to build the image locally and use it for development.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)

### Files

- `setup.sh`: Script for setting up the QEMU environment on your machine. Run this script ONCE to setup QEMU on your machine. JUST RUN THIS ONCE!
- `Dockerfile.dev`: Dockerfile for building the SUAS software in a Docker container for development purposes. This is the easiest way to get the software running on your machine if you are having trouble with the installation instructions.
- `push_dev.sh`: Script for building and pushing `Dockerfile.dev` image to Docker Hub. Only use if you added something that requires changes in the dev environment. ALWAYS USE A PROPER TAG.

## Files for running on the actual hardware in competition

Tried to use this on the drone to run the software in a container, but it didn't work.

- `Dockerfile`: Experimental Dockerfile for running the SUAS software in a Docker container, but was never used in competition because it didnt really work.
- `docker-compose.yml`: Docker Compose file for that describes how you would run the `Dockerfile` on the actual hardware in competition. This file was never used in competition because the `Dockerfile` didn't work.
