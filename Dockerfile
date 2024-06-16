FROM python:3.8
RUN mkdir /home/build
COPY . /home/build
WORKDIR /home/build

USER root

RUN export DEBIAN_FRONTEND=noninteractive && \
    export ACCEPT_EULA=Y && \
    apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y --no-install-recommends \
      net-tools \
      xcb \ 
      git \
      ffmpeg \
      libglib2.0-0 \
      libgl1-mesa-glx && \
    apt-get -y clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir wheel
RUN pip install --no-cache-dir ./model
RUN pip install --no-cache-dir .

EXPOSE 8080
ENTRYPOINT [ "python", "backend/lipread/manage.py", "runserver", "0.0.0.0:8080" ]