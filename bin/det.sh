#!/bin/bash
cd "$(dirname "$0")"

cd ..

mkdir -p log


MODEL_NAMES=(
  5cls_resnet50_unfreeze

  5cls_resnet50
  #5cls_resnet34

#  5cls_densenet201
#  5cls_densenet161
#
#  2cls_resnet34
#  2cls_resnet50

)


##################################################################################
# inference all models
##################################################################################
MODEL_NAME='default'
for fold in {0..4}
do

  CUDA_VISIBLE_DEVICES=$1 && python -u customer/detector.py main with conf_name=$MODEL_NAME fold=$fold  >> ./log/"$(hostname)"_d_$1.log 2>&1
  #python -u core/train_lgb.py train -1 $fold >> log/lgb_m_$fold.log 2>&1
done


#CUDA_VISIBLE_DEVICES=3  &&   python train.py --config resources/train_config_ce.yaml



