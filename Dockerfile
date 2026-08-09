FROM ubuntu:16.04

WORKDIR /
RUN apt-get update
RUN apt-get install -y git gcc g++ make
RUN apt-get install -y zlib1g zlib1g-dev
RUN apt-get install -y libssl-dev libncurses5-dev libsqlite3-dev libreadline-dev libtk8.6 libgdm-dev libdb4o-cil-dev libpcap-dev
RUN apt-get install -y libsasl2-dev libldap2-dev libssl-dev

RUN git clone https://github.com/deadsnakes/python3.6
WORKDIR /python3.6
RUN ./configure --enable-optimizations
RUN make
RUN make test
RUN make install

RUN python3.6 -m pip install pip --upgrade
RUN python3.6 -m pip install wheel

RUN pip3 install pyOpenSSL
COPY requirements.txt .
RUN pip3 install -r requirements.txt
