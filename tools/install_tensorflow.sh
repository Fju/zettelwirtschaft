#!/bin/bash
sudo apt-get install python-pip python-dev python-virtualenv
virtualenv --system-site-packages ../tensorflow
source ../tensorflow/bin/activate
easy_install -U pip
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.6.0-cp27-none-linux_x86_64.whl
deactivate
