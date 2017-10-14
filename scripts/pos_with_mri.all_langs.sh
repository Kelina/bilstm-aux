#!/bin/bash

langs=('gle' 'lit' 'uk')

cd ..

mkdir models
mkdir models/pos_with_mri

for lang in "${langs[@]}";
do
  python3 src/bilty.py --dynet-mem 8000 --train data/pos/${lang}-ud-train.conllu.pos data/mri/${lang}-train --dev data/pos/${lang}-ud-dev.conllu.pos --save models/pos_with_mri/${lang} --iters 50 --pred_layer 1 1 --task_types original mri --trainer adam --dynet-autobatch 1 --minibatch-size 20 >& logs/${lang}-pos_with_mri &
done
