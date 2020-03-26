# DVS Optic Flow

This is a SCNN model for optical flow using DVS signal.

## Requirements

To run sim session, you need to install:

* [ROS](http://wiki.ros.org/kinetic/Installation/Ubuntu) (kinetic tested)
* [rpg_esim](https://github.com/uzh-rpg/rpg_esim.git)
* gnome_terminal
* zsh

You can follow the instruction [here](https://www.everness.me/tech/事件相机模拟器rpg_esim安装指北/).

## Usage

### Simulation

To run sim session, you need to specify your dataset:

```python
# sim/sim.py
root = "/path/to/dataset"
```

Then you can simply run:

```bash
python sim/sim.py
```

### Dataset

To use the dataloader, you need to create a list file named `list.csv` under your `training` and `test` path. The list file is modified as:

```
set_name,frames

Example:
alley_1,50
alley_2,50
```

Ensure your root path like this

```
root
|-training
  |-final	//your dvs file here
  |-flow	//your flow file here
  |-list.csv
|-test
  |-final	//your dvs file here
  |-flow	//your flow file here
  |-list.csv
```

Then you can call dataloader

```python
dataset = DVSFlowDataset(root, train=True, preload=False)
```



