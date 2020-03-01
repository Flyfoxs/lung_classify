#!/bin/bash
cd "$(dirname "$0")"

cd ..

mkdir -p log



##################################################################################
# inference all models
##################################################################################
#for MODEL_NAME in ${MODEL_NAMES[@]}; do



search_dir=configs

for version in {0..4}
do
  for fold in {0..0}
  do
    for conf_file in "$search_dir"/ens_*.yaml
    do
        for lock_layer in {0..0}
        do
          conf_file="$(basename $conf_file .yaml)"
          CUDA_VISIBLE_DEVICES=$1 && python -u customer/classify.py main with conf_name=$conf_file fold=$fold lock_layer=$lock_layer version=c$version >> ./log/ens_"$(hostname)"_$1.log 2>&1
          #python -u core/train_lgb.py train -1 $fold >> log/lgb_m_$fold.log 2>&1
        done
    done
  done
done