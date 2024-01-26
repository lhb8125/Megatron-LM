#!/bin/bash

set -u

# export MODEL_SIZE="15b" PP="1"
export MODEL_SIZE="340b" PP="2"
# export MODEL_SIZE="340b" PP="4"

bash ./convert_340b.sh

