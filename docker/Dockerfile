FROM tensorflow/tensorflow:2.4.0-gpu
LABEL maintainer="amf85@cam.ac.uk"
WORKDIR ~/

RUN pip3 install --upgrade pip && pip install tensorflow-datasets matplotlib
#RUN pip3 install pkgconfig Cython numpy dill gym networkx pandas
#RUN pip3 install wget
#RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN apt-get update -y
RUN apt-get update -y && apt update -y && apt install -y graphviz graphviz-dev
#RUN pip install pygraphviz
RUN pip3 install pygraphviz pydot
#RUN pip3 install -Uq apache_beam
#RUN pip3 install -q tensorflow-transform

COPY cache_data.py cache_data.py

RUN python cache_data.py

CMD ['bash']