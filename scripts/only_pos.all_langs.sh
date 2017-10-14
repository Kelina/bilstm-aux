#!/bin/bash

add_params=$1
#langs=('ar' 'de' 'da' 'en' 'es' 'ru')
langs=('lit' 'gle' 'uk')

cd ..
mkdir models
mkdir models/only_pos

for lang in "${langs[@]}";
do
  python3 src/bilty.py --dynet-mem 8000 --train data/pos/${lang}-ud-train.conllu.pos --dev data/pos/${lang}-ud-dev.conllu.pos  --iters 50 --pred_layer 1 --task_types original --trainer adam --dynet-autobatch 1 --minibatch-size 20 --save models/only_pos/${lang} >& logs/${lang}-only_pos &
done
