# MobileNet-SSD and MobileNetV2-SSD/SSDLite with PyTorch

Object Detection with MobileNet-SSD, MobileNetV2-SSD/SSDLite on VOC, BDD100K Datasets.

<!-- ## Results
1. Detection

<img src="readme_images/detection_105e.jpg" width="1200">
-->

## Dependencies
- Python 3.8
- OpenCV
- PyTorch
- tensorboard
- tqdm

## Usage
1. Get dataset
```bashrc
$ mkdir ubi
$ cd ubi
$ git clone https://github.com/pete710592/UBI_Dataset.git
```
2. Get model
```bashrc
$ cd ..
$ git clone https://github.com/yue-723/UBI_SSD.git
```
3. Split dataset
```
notebook: /UBI_SSD/Dataset/generate_Data.ipynb
```

## Train
1. Train
```bashrc
$ python train.py --datasets <DATASET_PATH> --validation_dataset <VALIDSET_PATH> --net <NET_TYPE> --batch_size 64 --num_epochs 200 --scheduler cosine --lr 0.01 --t_max 200
```
2. Train pretrained-model
```bashrc
$ python train.py --datasets <DATASET_PATH> --validation_dataset <VALIDSET_PATH> --net <NET_TYPE> --pretrained_ssd models/${pretrained-model}.pth --batch_size 64 --num_epochs 200 --scheduler cosine --lr 0.01 --t_max 200
```

## Resume Training
```bashrc
$ python train.py --datasets <DATASET_PATH> --validation_dataset <VALIDSET_PATH> --net <NET_TYPE> --resume models/${trained-model-name}.pth --batch_size 64 --num_epochs 200 --last_epoch ${last_epoch} --scheduler cosine --lr 0.001 --t_max 100 --debug_steps 10
```

## Test
1. Test on image
```bashrc
$ python test.py <IMG_PATH> <MODEL_WEIGHT_PATH> <MODEL_TYPE>
```

## Example args

1. Train pretrained-model
```bashrc
$ python train.py --datasets /Data/train/ --validation_dataset /Data/val/ --net mb2-ssd-lite --pretrained_ssd models/mb2-ssd-lite-net.pth --batch_size 64 --num_epochs 200 --scheduler cosine --lr 0.001 --t_max 200
```
2. Resume
```bashrc
$ python train.py --datasets /Data/train/ --validation_dataset /Data/val/ --net mb2-ssd-lite --resume models/best.pth --batch_size 64 --num_epochs 20 --scheduler cosine --lr 0.001 --t_max 200
```
3. Test
```bashrc
$ python test.py Data/test/images/Cityscape_01.mp4_00011.png /models/best.pth mb2-ssd-lite
```

## Export model
```bashrc
$ python export.py
```

## Convert to .rknn model
```bashrc
$ python convert2rknn.py
```

## References
- https://github.com/qfgaohao/pytorch-ssd
- https://github.com/leeesangwon/bdd100k_to_VOC
- https://github.com/tranleanh/mobilenets-ssd-pytorch
