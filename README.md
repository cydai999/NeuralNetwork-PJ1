# README

本项目实现了简单的三层全连接神经网络及卷积神经网络，在MNIST数据集上的测试精度可以达到0.98。这篇文档主要介绍了如何训练及测试该模型，包含前期准备，训练，测试及可视化模型权重四个部分。

## Set up

首先需要将模型代码下载到本地：

```cmd
git clone https://github.com/cydai999/NeuralNetwork-PJ1
```

之后从https://drive.google.com/drive/folders/1GdLbs1mjBJ2-k6Kd5bLahiI7nGf_UP-v?usp=drive_link下载模型权重与数据集文件夹，解压缩后将其放在项目根目录下。相对路径位置应如下所示：

```plaintext
PJ1
├── dataset 
│   └── MNIST
├── saved_models
│   ├── CNN
│   └── MLP
└── code
    ├── mynn
    ├── draw_tools
    ├── train.py
    ├── train_CNN.py
    ├── test.py
    ├── hyperparam_search.py
    └── visual_weight.py
```



## Train

### 训练MLP模型

训练模型的过程非常简单，只需在**根目录**下打开终端，输入如下指令，即可以以默认配置进行训练：

```
python code/train.py
```

除此之外，支持自定义模型超参数，以下列出一些可选参数及说明：

```
--hidden_size(-hs): 隐藏层维度，默认值1000
--act_func(-a): 激活函数类型，可在'Sigmoid', 'ReLU', 'LeakyReLU'中选择，默认值'LeakyReLU'
--weight_decay_param(-wd): 权重衰减系数，默认值1e-3
--init_lr(-lr): 初始学习率，默认值0.1
--step_size(-s): 学习率衰减周期，默认值2（2个epoch）
--gamma(-g): 学习率衰减系数，默认值0.1
--batch_size(-bs): 每个批次的样本量，默认值32
--epoch(-e): 遍历轮数，默认值5
--log_iter(-l): 打印loss和accuracy周期，默认值100
--early_stop(-es): 是否使用早停法，默认值False
--patience(-p): 早停法阈值，默认值2
--data_augmentation(-da): 是否使用数据增强，默认值False
```

示例：

比如，想要以1e-2的学习率训练10个epoch，可以输入如下指令：

```
python code/train.py -lr 1e-2 -e 10
```

### 训练CNN模型

训练CNN的过程同样非常简单，只需在**根目录**下打开终端，输入如下指令，即可以以默认配置进行训练：

```
python code/train_CNN.py
```

除此之外，支持自定义模型超参数，以下列出一些可选参数及说明：

```
--channel_list(-ch): 卷积层通道数，默认值[6, 16]
--kernel_list(-ks): 卷积核边长，默认值[5, 5]（需与channel_list长度一致）
--hidden_size_list(-hs): 隐藏层维度，默认值[128, 64]
--act_func(-a): 激活函数类型，可在'Sigmoid', 'ReLU', 'LeakyReLU'中选择，默认值'LeakyReLU'
--weight_decay_param(-wd): 权重衰减系数，默认值1e-3
--linear_weight_decay_param(-lwd): 线性层权重衰减系数，默认值1e-3
--init_lr(-lr): 初始学习率，默认值0.01
--step_size(-s): 学习率衰减周期，默认值5（5个epoch）
--gamma(-g): 学习率衰减系数，默认值0.1
--batch_size(-bs): 每个批次的样本量，默认值32
--epoch(-e): 遍历轮数，默认值5
--log_iter(-l): 打印loss和accuracy周期，默认值100
--early_stop(-es): 是否使用早停法，默认值False
--patience(-p): 早停法阈值，默认值2
```



## Test

若想要测试训练好的MLP模型在测试集上的表现，可以在终端中输入：

```
python code/test.py
```

假如想测试训练好的CNN模型表现，也可以通过`-p`指定模型路径：

```
python code/test.py -p ./saved_models/CNN/best_model/models/best_model.pickle
```



## Visual

若想要可视化训练好的MLP模型模型参数，可以在终端输入：

```
python code/visual_weight.py
```

同样，假如想可视化训练好的CNN的模型参数，也可以通过`-p`指定模型路径：

```
python code/visual_weight.py -p ./saved_models/CNN/best_model/models/best_model.pickle
```



