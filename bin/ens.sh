#!/bin/bash
cd "$(dirname "$0")"

cd ..

mkdir -p log
pwd
search_dir=configs
for conf_file in "$search_dir"/ens*.yaml
do
  for fold in {0..4}
  do
    CUDA_VISIBLE_DEVICES=$1 && python -u customer/classify_ensemble.py main with conf_name=$conf_file fold=$fold version=r10  >> ./log/ens_"$(hostname)"_$1.log 2>&1
    #python -u core/train_lgb.py train -1 $fold >> log/lgb_m_$fold.log 2>&1
  done
done

#CUDA_VISIBLE_DEVICES=3  &&   python train.py --config resources/train_config_ce.yaml



