# Dataloader for FlyChairs
## Usage
1. 更改`img2dvs.py`中的相关参数并运行，对数据集进行预处理。
    ```python
    # 数据集根目录
    root = "D:\\Dataset\\train"
    # 数据集总量
    total = 10
    # 采样时间窗个数
    time_window = 5
    ```
   运行后会在根目录下按比例输出`train`与`test`两个数据集。

2. 参照`dataloader_test.py`中的方式调用数据集。
    ```python
   dataset = DVSDataset(root, size, time_window=5, train=True)
    ```
   其中`root`，`time_window`须和`img2dvs`中设置量保持一致，`size`为数据集下对应的文件数。