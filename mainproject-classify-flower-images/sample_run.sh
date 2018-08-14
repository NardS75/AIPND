#!/bin/bash
python train.py flowers --save_dir checkpoints/ --arch vgg11 --hidden_units 1024,1024 --gpu
python train.py flowers --save_dir checkpoints/ --arch vgg19 --hidden_units 1024,1024 --gpu
python train.py flowers --save_dir checkpoints/ --arch vgg16_bn --hidden_units 1024,1024 --gpu
python train.py flowers --save_dir checkpoints/ --arch densenet121 --gpu
python train.py flowers --save_dir checkpoints/ --arch densenet201 --gpu
python train.py flowers --save_dir checkpoints/ --arch alexnet --hidden_units 4096,4096 --gpu



python predict.py test_img.jpg checkpoints/cp_densenet121_e7_lr0.05.pth --top_k 5 --category_names cat_to_name.json



