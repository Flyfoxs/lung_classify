#!/bin/bash
cd "$(dirname "$0")"

cd ..

mkdir -p log


MODEL_NAMES=(
#  5cls_resnet50_unfreeze

  5cls_resnet50

#  2cls_resnet50

  5cls_densenet201
  5cls_densenet161
#
#  2cls_resnet34
#  2cls_resnet50

)


##################################################################################
# inference all models
##################################################################################
for MODEL_NAME in ${MODEL_NAMES[@]}; do
  for fold in {0..4}
  do

    CUDA_VISIBLE_DEVICES=$1 && python -u customer/reg.py main with conf_name=$MODEL_NAME fold=$fold  >> ./log/reg_"$(hostname)"_$1.log 2>&1
    #python -u core/train_lgb.py train -1 $fold >> log/lgb_m_$fold.log 2>&1
  done
done

#CUDA_VISIBLE_DEVICES=3  &&   python train.py --config resources/train_config_ce.yaml



