FROM debian:latest

RUN apt-get update && apt-get install -y python3 python3-pip git python3-venv

RUN python3 -m venv /venv

ENV PATH="/venv/bin:$PATH"

RUN git clone https://github.com/zooai/axolotl.git /axolotl

RUN cd /axolotl && pip install -e .
