FROM jupyter/minimal-notebook
MAINTAINER Sabrina Steinert
COPY ./code /assignment4
WORKDIR /assignment4
RUN pip install -r requirements.txt
