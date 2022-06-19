#!/bin/bash

tc qdisc add dev lo root netem rate 2Mbit

clang -O2 -g -Wall -target bpf -c xdp_drop.c -o xdp_drop.o
ip link set dev lo xdp obj xdp_drop.o sec xdp_drop

sleep $1

ip link set dev lo xdp off

tc qdisc del dev lo root