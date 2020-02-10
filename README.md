# SNN_starter

## Overview

SNN moudle rebuild of *A Low Power, Fully Event-Based Gesture Recognition System* using [yjwu17/BP-for-SpikingNN](https://github.com/yjwu17/BP-for-SpikingNN).

## Usage

Download the dataset from [here](http://research.ibm.com/dvsgesture/) and extract as ```dataset```.

Ensure that all files (eg. ```*.adept```, ```*.csv```, ```trails_to_train.txt``` and ```trails_to_test.txt```) are **NOT** in any directory other than ```./dataset```.

```bash
python train.py
```

Change ```preload = True``` to use formatted data, data is saved at ```root/*.data```

## Tips

* snn project for gesture recognition trial文件中将测试数据集只用了四个，供我们本机运行

