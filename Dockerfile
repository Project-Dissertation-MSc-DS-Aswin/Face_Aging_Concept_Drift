FROM ubuntu:20.04

RUN apt-get update && apt-get install python3 python3-pip wget unzip git curl -y

WORKDIR /home/project

ARG DEBIAN_FRONTEND=noninteractive
RUN set -ex && apt-get install build-essential python3-distutils python3-apt -y

RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install python3.7 -y
COPY get-pip.py /home/project/get-pip.py
RUN apt-get install python3.7-distutils -y
RUN python3.7 get-pip.py
COPY ./requirements.txt /home/project
RUN apt-get install python3.7-dev -y
RUN pip3.7 install -r requirements.txt
RUN apt-get install libgl1-mesa-glx -y

RUN apt install lsb-release
RUN curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg

RUN echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list

RUN apt-get update
RUN apt-get install redis -y

RUN pip3.7 install fastapi uvicorn pydantic python-socketio eventlet 
RUN pip3.7 install opencv-python
RUN pip3.7 install redis aioredis
RUN pip3.7 install matplotlib

RUN wget https://project-dissertation.s3.eu-west-2.amazonaws.com/facenet_keras.h5 -P /home/project/src/models
RUN wget https://project-dissertation.s3.eu-west-2.amazonaws.com/vit_face_recognition_model.h5 -P /home/project/src/models

RUN unzip /home/project/src/models/all_models.zip /src/models
RUN cp -Rf /home/project/src/models/all_models/*.pkl /home/project/src/models
RUN unzip /home/project/src/models/16.07.2022_two_classifiers.zip /src/models
RUN cp -Rf /home/project/src/models/16.07.2022_two_classifiers/*.pkl /home/project/src/models

WORKDIR /home/project/

CMD ["/bin/bash"]
