FROM openvino/ubuntu18_dev

WORKDIR /home/project/

USER root

RUN apt-get install python3 python3-dev python3-pip -y

COPY ./requirements.txt ./
RUN pip3 install -r requirements.txt

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install python3.7 -y

CMD ["/bin/bash"]