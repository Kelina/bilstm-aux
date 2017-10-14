#!/bin/bash

add_params=$1
langs=('gle' 'lit' 'uk')

cd ..
#python3 src/bilty.py --dynet-mem 1500 --train data/mri/danish-train --dev data/mri/danish-dev  --iters 50 --pred_layer 1 --task_types mri

mkdir models
mkdir models/only_mri

for lang in "${langs[@]}";
do
  python3 src/bilty.py --dynet-mem 4000 --train data/mri/${lang}-train --dev data/mri/${lang}-dev --save models/only_mri/${lang} --iters 50 --pred_layer 1 --task_types mri --trainer adam --dynet-autobatch 1 --minibatch-size 20 >& logs/${lang}-only_mri &
done
