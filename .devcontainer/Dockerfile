FROM osrf/ros:noetic-desktop-full

SHELL ["/bin/bash", "-c"]

# Update all packages
RUN apt update && apt upgrade -y

# Install packages
RUN apt install -y \
	# already present in the base image
	# build-essential \
	# cmake \
	# libeigen3-dev \
	# libopencv-dev \
	# libprotobuf-dev \
	# libprotoc-dev \
	# libxml2-utils \
	# protobuf-compiler \
	# python3 \
	# python3-empy \
	# python3-numpy \
	# python3-rosdep \
	# python3-rosinstall \
	# python3-rospkg \
	# python3-wstool \
	# ros-noetic-rqt-common-plugins \
	# libgazebo11 \
	# extra packages
	curl \
	genromfs \
	git \
	gstreamer1.0-alsa \
	gstreamer1.0-doc \
	gstreamer1.0-gl \
	gstreamer1.0-gtk3 \
	gstreamer1.0-libav \
	gstreamer1.0-plugins-bad \
	gstreamer1.0-plugins-base \
	gstreamer1.0-plugins-good \
	gstreamer1.0-plugins-ugly \
	gstreamer1.0-pulseaudio \
	gstreamer1.0-qt5 \
	gstreamer1.0-tools \
	gstreamer1.0-x \
	libgstreamer-plugins-base1.0-dev \
	libgstreamer1.0-dev \
	ninja-build \
	python3-pip \
	qtcreator \
	wget \
	zip \
	# ros stuff, where install through apt is recommended
	python3-rosinstall-generator \
	# mavros
	ros-noetic-mavros \
	ros-noetic-mavros-extras \
	ros-noetic-rqt 

# not really needed, should be installed via pip?
# python-argparse \
# python-dev \
# python-numpy \
# python-yaml 
# python3-jinja2 \
# python3-toml \

# not installed by default
# apt-get remove modemmanager -y

# Install pip packages
RUN pip install \
	argparse \
	empy \
	jinja2 \
	jsonschema \
	numpy \
	packaging \
	pandas \
	pyserial \
	pyulog \
	pyyaml \
	toml 

# oh god why do we have python2.7 shit
# RUN pip install future 

# Rosdep update
RUN rosdep update

# Mavlink
WORKDIR /usr/include
RUN git clone https://github.com/mavlink/c_library_v2.git --recursive
RUN mv /usr/include/c_library_v2/* /usr/include
RUN rm -r /usr/include/c_library_v2


# Gazebo
# the previous script installed libgazebo11 explicitly, wihch is already installed
# the curl command just used apt to install libgazebo11-dev and gazebo11
# both of which are already installed in the image

# PX4
WORKDIR /root
RUN git clone https://github.com/PX4/PX4-Autopilot.git --recursive
RUN ./PX4-Autopilot/Tools/setup/ubuntu.sh
# PX4 sitl

# qgroundcontrol


# MavROS
# already installed earlier with apt


# Source the ROS entrypoint file
RUN source /opt/ros/$ROS_DISTRO/setup.bash



