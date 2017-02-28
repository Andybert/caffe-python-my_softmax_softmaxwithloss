#!/usr/bin/env sh
set -e

export PYTHONPATH="/mnt/68FC8564543F417E/caffe/caffe-master/python:$PYTHONPATH"
export PYTHONPATH="/mnt/68FC8564543F417E/caffe/caffe-master/python/Softmax:$PYTHONPATH"

./build/tools/caffe train --solver=examples/mymnist/lenet_solver.prototxt $@
