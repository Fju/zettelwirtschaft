#!/bin/bash
sudo apt-get install python-pip python-dev python-virtualenv
virtualenv ../tensorflow
source ../tensorflow/bin/activate
pip install --upgrade pip
pip install --upgrade tensorflow pandas sklearn
deactivate
