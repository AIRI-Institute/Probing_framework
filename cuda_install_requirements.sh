#!/bin/bash

read -rp "Enter cuda version (e.g. '101' or 'cpu' to avoid installing cuda support): " cuda_version

if [ "$cuda_version" == "cpu" ]; then
  pip3 install -r requirements.txt
  pip install torch==1.12 --find-links https://download.pytorch.org/whl/cpu/torch_stable.html
else
	read -rp "Enter torch version (>= 1.12.0): " torch_version
  pip3 install torch==${torch_version}+cu${cuda_version} -f https://download.pytorch.org/whl/torch_stable.html
  pip3 install -r requirements.txt
fi
