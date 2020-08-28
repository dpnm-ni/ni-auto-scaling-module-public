#!/bin/bash

# Get library to use functions
source library

isSet_hostname # Solve the hostname warning
check_package=`dpkg -l | grep traceroute | wc -l`

if [[ $check_package -lt 2 ]]; then
  sudo apt-get install traceroute # Install traceroute
fi

if [[ $1 == "-h"  ]]; then
  echo "[Help] Please enter the command: ./test_traceRoute.sh {target_server}"
  echo "[Help] example: ./test_traceRoute.sh 8.8.8.8"
else
  if [[ ! -z $1 ]]; then
    traceroute $1
  else
    echo "[Error] Please enter a target_server"
  fi
fi
