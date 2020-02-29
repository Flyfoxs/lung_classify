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
    for conf_file in "$search_dir"/5cls_*.yaml
    do
      conf_file="$(basename $conf_file .yaml)"
      CUDA_VISIBLE_DEVICES=$1 && python -u customer/classify.py main with conf_name=$conf_file fold=$fold version=c$version >> ./log/ens_"$(hostname)"_$1.log 2>&1
      #python -u core/train_lgb.py train -1 $fold >> log/lgb_m_$fold.log 2>&1
    done
  done
done