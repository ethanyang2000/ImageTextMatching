1. 代码各个部分：
   + Args类代表的是实验的各种参数。
   + Accuracy实现的是算法的metric。
   + BasicBlock，ResNet，ResNet18是图片特征提取器的结构。
   + myDataset是重载的数据类，实现了一些数据清洗和读取的算法。
   + trainer是主要的类，包括数据读取和训练、推理的代码。调用train和inference可以根据定义的路径开始训练或者推理。
   + SupConLoss实现的是监督对比学习的loss。
   + MatchingNet是匹配器的主要结构。
   + InitNet是封装的特征提取器，包括ResNet和BERT。

2. 代码主要环境：
   + python 3.8.0
   + torch 1.7.0+cu110
   + torchvision 0.8.1
   + transformers 4.20.0
   + numpy 1.21.5

3. 数据集位置：数据集的medium目录和project.py文件并列放置，在同一个文件夹的根目录。

4. 实验参数
   + hiddensize=512
   + batch_size = 256
   + extra_crop = True
   + crop = False
   + normalize='batch'
   + use_attn = True
   + use_gru = True
   + fix_resent=True
   + train_resnet = True
   + train_both = False
推理时需要设置inference为任意的字符串。具体的参数可以参考报告中附上的网址。
