FROM python:3.10-bullseye

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      wget \
      python3-tk && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir ./images

RUN pip3 install numpy flask opencv-python opencv-contrib-python

COPY . .

CMD ["python3", "-m" , "flask", "run", "--host=0.0.0.0"]