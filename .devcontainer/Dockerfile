ARG VARIANT="3.8"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

RUN pip install --upgrade pip \
  && pip install gym fire tensorflow matplotlib

RUN pip install pybullet

RUN pip install tensorflow-probability

RUN pip install tf-agents