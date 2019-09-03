# CAFFE with PyThon Layers for darknet

### Data preparation
> Prepare data lsit just like the way of original darkenet
> No need to generate lmdbs as we directly use raw images in the data list to train


### Train tiny yolo v2
set the environment
```
export PYTHONPATH=../../lib/:../../../caffe/python/:$PYTHONPATH
```

train the detection
```
cd Root_Repo/yolo/models/tiny-yolo-v2/
sh train_tiny.sh
```


### Reference

> You Only Look Once: Unified, Real-Time Object detection http://arxiv.org/abs/1506.02640

> YOLO9000: Better, Faster, Stronger https://arxiv.org/abs/1612.08242





## Version
> Both v1 & v2 supported
