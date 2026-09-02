#!/bin/bash
(curl -sN --unix-socket /var/run/docker.sock "http://localhost/containers/02a5d53cf19783a6d8b2b2a64ebdccaa5ae637339d8e21ca2ebb5faecda1dc82/logs?follow=true&stdout=true&stderr=true&tail=all" | python3 /work/tb_bridge.py) &
exec python3 -m tensorboard.main --logdir /work/out/tb --port 6006
