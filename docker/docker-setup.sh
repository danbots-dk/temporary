#!/bin/bash

# x11 forwarding for viewing cv2 graphs etc.
export DISPLAY=:1.0
xhost +local:root



# if [[ $( docker ps | awk '{if(NR>1) print $NF}' ) == 'inference-dev' ]]; then 
#     echo "Stopping inference-dev container"
#     docker stop inference-dev
# fi
# echo "Pruning all containers"
# docker container prune

docker container stop inference-dev || echo container not running
docker container prune || echo container not pruned

docker run --gpus all  -d -t --name inference-dev \
-v /data:/data \
-v /danbots:/danbots \
-v /home/samir/Desktop:/home/samir/Desktop \
-v /home/samir/sal_github/docker/inference-dev-server:/home/samir/sal_github/docker/inference-dev-server \
-v /home/samir/Desktop/blender:/home/samir/Desktop/blender \
-v /var/run/dbus:/var/run/dbus \
-v /var/run/avahi-daemon/socket:/var/run/avahi-daemon/socket \
--restart unless-stopped \
--env="DISPLAY" \
--env="QT_X11_NO_MITSHM=1" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
--volume="/etc/localtime:/etc/localtime:ro" \
--volume="/etc/timezone:/etc/timezone:ro" \
--net=host \
--user 1000:1000 \
 inference-dev:1
#docker run --gpus all -v $1:/home --rm -it --name lambdastack lambdastack:1 bash   
