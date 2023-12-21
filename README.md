# LeNet5 卷积神经网络与脉冲神经元网络的转换

我们参考Braincog框架中提供的神经元节点模型以及转换器，将经典的LeNet5卷积神经网络（CNN）转换为更为高效的脉冲神经元网络（SNN）模型。

## 成员分工：
- **史显宇**：代码、CNN思路构建
- **孙舒同**：代码
- **李泽一**：代码
- **殷笑天**：整理报告、仓库构建
- **张 &nbsp; &nbsp;沁**：整理报告、代码上传

## MNIST 数据集介绍
MNIST数据集是手写数字识别的标准数据集，包含60000张训练图像和10000张测试图像，每张图像大小为28x28像素。
![MNIST 数据集示例](https://raw.githubusercontent.com/zq111724/pic/main/2.png)
我们使用pytorch的torchvision工具集下载MNIST的训练和测试图片，数据集的图片以字节形式存储，可直接使用`torch.utils.data.DataLoader`进行加载。

## LeNet5 网络介绍
在本次项目中，我们使用LeNet5实现手写数字识别目标。下面是对LeNet5的详细介绍：
![LeNet5 架构](https://raw.githubusercontent.com/zq111724/pic/main/1.png)
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

## 转换细节
1. **SpikeNode类**：模拟神经元真实的脉冲行为。
2. **fuse函数**：对卷积层权重进行调整，融合批标准化层的信息。
3. **SmaxPool类**：自定义脉冲神经元的最大池化层。
4. **fuse_norm_replace函数**：对LeNet5的卷积层、批标准化层等进行处理和替换。

## 转换操作
- 运行`lenet5.py`，利用传统LeNet5网络训练MNIST数据集。
- 运行`lenet_convert.py`，加载权重并实例化转换器，对脉冲神经网络进行推理。

## 代码解释

### lenet-convert.py
- **LeNet5模型**：定义标准的LeNet-5架构。
- **SpikeNode和SMaxPool**：模拟脉冲神经元行为和脉冲最大池化操作。
- **fuse_norm_replace函数**：融合卷积层和批标准化层，替换ReLU激活为SpikeNode。
- **clean_mem_spike函数**：在训练过程中更换批次时清除内存和脉冲。
- **evaluate_snn函数**：评估SNN在MNIST测试集上的准确性。

### lenet5.py
- **LeNet5模型**：定义LeNet-5模型架构。
- **hook函数**：在每个前向传播中记录每层的输出。
- **train函数**：进行模型训练，包括梯度反向传播和参数更新。
- **evaluate_accuracy函数**：评估模型在测试集上的准确性。
