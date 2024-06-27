# Docker-tensorflow
Contents of this repo is as follows:
1. docker-install.sh
    - This script contains the commands needed to install docker and the necesary nvidia complementary libraries. Need to be run with sudo and a reboot afterwards is a good idea.

2. docker-build.sh
    - This script contains the commands needed to build the tensorflow image we use for the containers. This should only run once in order to build the       image, however if any changes are made the Dockerfile_tensorflow file (our build recipe) run the script again. Run without sudo.

3. docker-setup.sh
    - This script start a container based on the previosly created tensorflow image. This should only run once, after that docker will start the container with the specified arguments on boot.

## How to use

1. ./docker-install.sh
2. ./docker-build.sh
3. ./docker-setup.sh

Enter vs code and install an extension named dev containers by microsoft. Once installed hit ctrl+shift+p and enter "attach to running container" in the prompt at select the tensorflow container. A new vs code window that works inside the container will open.
# temporary
