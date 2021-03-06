#!/bin/bash
cd "$(dirname "$0")"

cd ..

mkdir -p log



##################################################################################
# inference all models
##################################################################################
#for MODEL_NAME in ${MODEL_NAMES[@]}; do



search_dir=configs
model_type=fastai
backbone=resnet34

for version in {0..4}
do
  for fold in {0..0}
  do
        for lock_layer in {0..9}
        do
          conf_file="$(basename $conf_file .yaml)"
          CUDA_VISIBLE_DEVICES=$1 && python -u customer/classify.py main with backbone=$backbone \
                                            model_type=$model_type \
                                            version=c$version fold=$fold lock_layer=$lock_layer >> ./log/ens_"$(hostname)"_$1.log 2>&1
          #python -u core/train_lgb.py train -1 $fold >> log/lgb_m_$fold.log 2>&1
        done
  done
done