#!/bin/bash
python train.py flowers --save_dir checkpoints/ --arch vgg16 --hidden_units 1024,1024 --bn --epochs 10
python train.py flowers --save_dir checkpoints/ --arch densenet201 --gpu --backend pytorch

python predict.py testimg.jpg checkpoints/cp_densenet121_e7_lr0.05.pth --top_k 5 --category_names cat_to_name.json
python predict.py test checkpoints/cp_resnet50_e10_lr0.05.h5 --category_names cat_to_name.json



