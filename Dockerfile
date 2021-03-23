FROM tensorflow/tensorflow:2.4.1-gpu

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libxext6 \
    libx11-6 \
  && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
  && apt-get install -y python-opengl \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
  && pip install gym fire matplotlib

RUN pip install pybullet

RUN pip install tf-nightly

RUN pip install tfp-nightly

RUN pip install tf-agents-nightly

#RUN apt-get update \
#  && apt-get install -y -qq --no-install-recommends \
#    git \
#  && rm -rf /var/lib/apt/lists/*

#RUN pip install git+https://github.com/pedrogasg/agents@feature/saving-with-keras

WORKDIR /gym

COPY ./src .
#CMD [ "python", "./random_policy.py" ]
CMD [ "python", "./run.py" ]