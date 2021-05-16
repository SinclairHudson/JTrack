# Ubuntu 16.04 base image
FROM ubuntu
#RUN apt-get update
#RUN add-apt-repository universe
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update
RUN apt-get -y install python3-pip python3-opencv
RUN apt-get -y install ffmpeg

RUN pip3 install filterpy

ADD *.py /

#CMD [ "python3", "./video.py" ]


