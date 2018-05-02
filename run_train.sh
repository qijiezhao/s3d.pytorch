#!/usr/bin/env bash
#cd /local/MI/zqj/temporal_action_localization/TAL
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python trainval.py kinetics RGB train.list val.list --arch S3DG --epochs 32 --lr_steps 10 24 -d 16 -b 24 --lr 0.01 -p 20 -ef 1 -j 8 --snapshot_pref modelweights/I3D --flow_pref flow_
