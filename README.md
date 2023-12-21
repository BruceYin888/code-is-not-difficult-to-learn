# LeNet5 卷积神经网络与脉冲神经元网络的转换

我们参考Braincog框架中提供的神经元节点模型以及转换器，将经典的LeNet5卷积神经网络（CNN）转换为更为高效的脉冲神经元网络（SNN）模型，并在MNIST数据集上进行对比测试。

## 成员分工：
- **史显宇**：代码、CNN思路构建
- **孙舒同**：代码
- **李泽一**：代码、DeBug
- **殷笑天**：整理报告、仓库构建
- **张 &nbsp; &nbsp;沁**：整理报告、代码上传

## MNIST 数据集介绍
MNIST数据集是手写数字识别的标准数据集，包含60000张训练图像和10000张测试图像，每张图像大小为28x28像素。<br>
![MNIST 数据集示例](https://raw.githubusercontent.com/zq111724/pic/main/2.png)<br>
我们使用pytorch的torchvision工具集下载MNIST的训练和测试图片，数据集的图片以字节形式存储，可直接使用`torch.utils.data.DataLoader`进行加载。

## LeNet5 网络介绍
在本次项目中，我们使用LeNet5实现手写数字识别目标。下面是对LeNet5的详细介绍：<br>
![LeNet5 架构](https://raw.githubusercontent.com/zq111724/pic/main/1.png)<br>
LeNet5的基本结构包括7层网络结构（不含输入层），其中包括：
- 2个卷积层
- 2个降采样层（池化层）
- 2个全连接层
- 输出层

各层功能简述如下：
- **输入层**：接收大小为的手写数字图像，其中包括灰度值（0-255）。通常会对输入图像进行预处理，如归一化像素值。
- **卷积层C1**：包括6个卷积核，产生大小为的特征图（输出通道数为6）。
- **采样层S2**：采用最大池化操作，产生大小为的特征图（输出通道数为6）。
- **卷积层C3**：包括16个卷积核，产生大小为的特征图（输出通道数为16）。
- **采样层S4**：采用最大池化操作，产生大小为的特征图（输出通道数为16）。
- **全连接层C5**：将特征图拉成长度为400的向量，连接到一个含120个神经元的全连接层。
- **全连接层F6**：将120个神经元连接到84个神经元。
- **输出层**：由10个神经元组成，对应0-9的数字，输出分类结果。
该网络在MNIST数据集中表现优秀，准确率可达98%以上。

## 转换操作
- 运行`lenet5.py`，利用传统LeNet5网络训练MNIST数据集。
- 运行`lenet_convert.py`，加载权重并实例化转换器，对脉冲神经网络进行推理。

## 转换细节
- 我们根据生物神经元的电生理特性，利用神经元动作电位的发放（Spike脉冲）与细胞膜电位和阈值电压的关系，定义了SpikeNode类，用于模拟神经元真实的脉冲行为。相应的，我们还定义了动作电位和膜电位清除函数clean_mem_spike，
当更换训练样本批次时，通过调用该函数将SpikeNode的膜电位和Spike进行清零处理。SpikeNode节点用于替换掉传统CNN网络中的ReLu和LogSoftmax激活函数。<br><br>
- 我们定义了fuse函数，该函数对卷积层的权重进行调整，按照批标准化的参数进行缩放和偏置，以融合批标准化层的信息，并创建新的融合后的卷积层。这样做的目的是将卷积层和批标准化层合并，减少模型中的参数数量，提高网络的推算效率。<br><br>
- 我们自定义了脉冲神经元的最大池化层：SmaxPool类。该池化层可根据网络是否处于脉冲模式以及是否启用横向抑制机制进行不同的处理。在脉冲模式下，我们对累积的输入进行扩大操作，以强调输入中的最大值，并对其进行最大池化操作。<br>
随后将输入和经过扩大的累积输入相加，进行最大池化操作。最后将二者的差异作为输出。该设计可强调累积输入中的显著特征，同时考虑输入中的新信息。这更符合生物神经元中树突在时间上的刺激积累特性以及输入的重要性。
此外，该池化层可以通过横向抑制特性操作减少累积的输入信息，从而更好地模拟了脉冲神经元之间的相互影响。<br><br>
- 根据以上转换规则，我们通过fuse_norm_replace函数将LeNet5中的卷积层、批标准化层、池化层以及激活层分别进行处理和替换。并在该函数中对全连接层进行权重和偏置的归一化操作。<br>
我们利用hook钩子函数记录的网络在前向传播过程中当前层和前一层的最大激活值，将当前线性层的权重按照前一层最大激活值的比例进行归一化。相似的，我们还对偏置和阈值电压进行处理，以确保脉冲神经网络中权重、偏置和激活阈值按比例进行调整。

## 结论总结
- 传统CNN卷积神经网络准确率为0.994
- CNN转换为SNN之后网络准确率为0.9934052136479592
- 对比可知，准确率相差不大，得出该方法有效性。
![MNIST 数据集示例](https://raw.githubusercontent.com/zq111724/pic/main/3.jpg)<br>

## 代码概略
### lenet5.py
- **LeNet5模型**：定义了LeNet-5模型的架构，包括卷积层、批标准化层、ReLU激活函数以及全连接层。
- **hook函数**：用于在每个前向传播过程中记录每一层的输出中占据99%位置的值。这个函数是通过注册forward hook实现的。
- **train函数**：进行模型的训练，包括迭代训练集，计算损失，进行梯度反向传播，更新模型参数，并在每个epoch结束后评估在测试集上的准确性。
- **evaluate_accuracy函数**：评估模型在测试集上的准确性。
- **主要部分**：加载MNIST数据集，创建LeNet-5模型的实例，设置优化器和学习率调度器，然后调用train函数进行训练。最终，加载在测试集上表现最好的模型，并计算并输出其在测试集上的准确性。

### lenet-convert.py
- **LeNet5模型**：LeNet5类定义了标准的LeNet-5架构，用于图像分类。
- **SpikeNode和SMaxPool**：这些是自定义模块，模拟脉冲神经元的行为和脉冲最大池化操作。
- **fuse_norm_replace函数**：该函数用于融合卷积层和批标准化层，将ReLU激活替换为SpikeNode，并将最大池化层替换为SMaxPool。
- **clean_mem_spike函数**：该函数在训练过程中更换批次时清除上一批次的内存和脉冲。
- **evaluate_snn函数**：该函数在MNIST测试集上评估SNN的准确性，测量随时间步骤的准确性，并可选择绘制结果。
- **主要部分**：代码的主要部分加载MNIST数据集，创建LeNet5模型的实例，并在测试集上评估SNN的准确性。还使用fuse_norm_replace函数对网络进行了一些修改，并将SNN的准确性与等效的人工神经网络（ANN）进行了比较。

