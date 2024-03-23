#!/usr/bin/env bash
#
# Sets up docker so that it can be used to build docker images

printf "\n *** SETING UP DOCKER *** \n"
sudo systemctl start docker
printf "\n *** Logging out of any existing docker logins *** \n"
docker logout

printf "\n *** Please create an account on hub.docker.com and ask Mehmed to add you to the MAAV org *** \n"
sleep 1s

login_success=false
while [[ ${login_success} == false ]]; do
	read -r -p "Docker Hub Username: " username
	read -r -sp "Docker Hub Password: " pass

	if echo ${pass} | docker login --username ${username} --password-stdin; then
		printf "\n *** Docker login successful! *** \n"
		login_success=true
	else
		printf "\n *** Docker login unsuccessful, please try again... *** \n"
	fi
done

printf "\n *** Installing QEMU *** \n"
# https://docs.docker.com/build/building/multi-platform/#qemu
docker run --privileged --rm tonistiigi/binfmt --install all

builder_name="maav_builder"
printf "\n *** Creating a new builder %s *** \n" "$builder_name"
docker buildx create --name ${builder_name} --use --driver docker-container
docker buildx inspect --bootstrap

printf "\n ** If you saw no errors, CONGRATULATIONS! You can now use the scripts to build and push to dockerhub. *** \n"
