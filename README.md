# SNN_starter

## Overview

SNN moudle rebuild of *A Low Power, Fully Event-Based Gesture Recognition System* using [yjwu17/BP-for-SpikingNN](https://github.com/yjwu17/BP-for-SpikingNN).

## Usage

Download the dataset from [here](http://research.ibm.com/dvsgesture/) and extract as ```dataset```.
Raw dataset must be formatted before training and it may spend some time, you can download the formatted data from[here](https://cloud.tsinghua.edu.cn/d/a57761a2bc2945218eac/) for the first training.

Ensure that all files (eg. ```*.adept```, ```*.csv```, ```trails_to_train.txt``` and ```trails_to_test.txt```) are **NOT** in any directory other than ```./dataset```.

```bash
python train.py
```

Change ```preload = True``` to use formatted data, data is saved at ```root/*.data```

## Tips

* Change the trials_to_train.txt and trials_to_test.txt to control the size of training and test dataset.


