#!/bin/sh

# 1 -> speed
# 2 -> latency
# 3 -> packet loss

tc qdisc del dev lo root 2>/dev/null
tc class del dev lo root 2>/dev/null

tc qdisc add dev lo handle 1: root htb default 11
tc class add dev lo parent 1: classid 1:1 htb rate 1000Mbps
tc class add dev lo parent 1:1 classid 1:11 htb rate $1
tc qdisc add dev lo parent 1:11 handle 10: netem delay $2 loss $3 $4