# CAFFE with PyThon Layers for darknet

#### Get Started
```
git clone --recursive https://github.com/leon-liangwu/py-caffe-yolo.git
pip install -r requirements.txt
```

### Data preparation
> Prepare data lsit just like the way of original darkenet
> No need to generate lmdbs as we directly use raw images in the data list to train


### Train tiny yolo v1 or v2

for tiny yolo v1
```
cd Root_Repo
python scripts/train_net.py --solver=./yolo/tiny_yolo_v1/solver.prototxt [--weights=]
```

for tiny yolo v2
```
cd Root_Repo
python scripts/train_net.py --solver=./yolo/tiny_yolo_v2/solver.prototxt [--weights=]
```


### Reference

> You Only Look Once: Unified, Real-Time Object detection http://arxiv.org/abs/1506.02640

> YOLO9000: Better, Faster, Stronger https://arxiv.org/abs/1612.08242





## Version
> Both v1 & v2 supported
