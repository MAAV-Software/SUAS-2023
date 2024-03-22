#!/usr/bin/env bash
#
# Builds and pushes the Docker development environment to Docker Hub

set -eo pipefail

IMAGE_FILE="Dockerfile.dev"
DOCKERHUB_REMOTE="akshatdy/maav"
PLATFORMS="linux/arm64,linux/amd64"

show_help() {
	cat <<EOF
Usage: ${0##*/} [-t] ...
Push the Docker development environment to Docker Hub

  -h                       Display help and exit
  -t        TAGNAME        Tag name for the docker image
EOF
}
# Reset in case getopts has been used
OPTIND=1

# Set options
while getopts 'h?t:' opt; do
	case "$opt" in
	h | \?)
		show_help
		exit 0
		;;
	t)
		TAG=$OPTARG
		;;
	esac
done

# Discard the options
shift "$((OPTIND - 1))"

if [ -z "$TAG" ]; then
	printf "*** Error: A tag is required, use -h for help  ***\n" >&2
	exit 1
fi

# Debug message
printf "\n *** Building Docker image *** \n"
printf "* Image File: %s\n" "$IMAGE_FILE"
printf "* Platforms: %s\n" "$PLATFORMS"
printf "* Docker Hub Remote: %s\n" "$DOCKERHUB_REMOTE"
printf "* Tag Name: %s\n" "$TAG"

# Build the docker image
docker buildx build --push --platform $PLATFORMS --tag $DOCKERHUB_REMOTE:$TAG -f $IMAGE_FILE --progress=plain .
