FROM ubuntu:20.04

RUN apt-get update && apt-get install python3 python3-pip wget unzip git curl -y

WORKDIR /home/project

ARG DEBIAN_FRONTEND=noninteractive
RUN set -ex && apt-get install build-essential cmake -y

RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install python3.7 -y
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.7 get-pip.py
COPY ./requirements.txt /home/project
RUN apt-get install python3.7-dev -y
RUN pip3.7 install -r requirements.txt
RUN apt-get install libgl1-mesa-glx -y

WORKDIR /home/project/

CMD ["/bin/bash"]
