title: Softmax的Pytorch实现分类任务
author: Strive
tags:
  - 深度学习
  - Softmax
categories:
  - 深度学习
date: 2021-09-08 22:21:00
---
#### 一、Softmax回归的相关原理

#### 1、Softmax的引入

在机器学习和深度学习中，分类和回归是常见的两个问题。其中回归模型往往是通过输入一系列的特征，经过一定的处理输出一个预测值，如通过输入房屋的面积、房间数量等特征通过回归模型可以预测得到这间房屋的价格。而分类问题往往希望通过输入一些特征得到一个分类。如输入一张图像输入这张图像的类别(如是猫还是狗)。在实际的操作中，我们对硬性类别感兴趣，即属于何种类别。但我们往往得到的是软性类别，即得到属于每个类别的概率，概率最大的类别即为类别的预测值。得到这种概率的结果往往并不困难，只需要在简单的线性模型的输出层前套一层softmax函数即可实现。

#### 2、Softmax函数

由上所述，分类问题的重点是如何将模型的输出映射成概率。SoftMax函数的功能就是将多个神经元的输出映射到（0，1）的区间内，从而将这种输出看作概率。下图非常清晰的显示了Softmax的计算过程。

![upload successful](/images/softmax-1.png)
假设有一个数组V，Vi表示数组中第i个元素，则该元素的Softmax值为

![upload successful](/images/softmax-2.png)
softmax直白来说就是将原来神经元的输出3,1,-3套上softmax函数映射成为取值范围为(0,1)的值，而这些值的累和为1（满足概率的性质），那么我们就可以将它理解成概率，在最后选取输出结点的时候，我们就可以选取概率最大（也就是值对应最大的）结点，作为我们的预测结果进行输出。
#### 3、交叉熵损失函数

在分类问题中，尤其是在神经网络中，交叉熵函数非常常见。因为经常涉及到分类问题，需要计算各类别的概率，所以交叉熵损失函数又都是与sigmoid函数或者softmax函数成对出现。

比如用神经网络最后一层作为概率输出，一般最后一层神经网络的计算方式如下：
1.网络的最后一层得到每个类别的scores。
2.score与sigmoid函数或者softmax函数进行计算得到概率输出。
3.第二步得到的类别概率与真实类别标签的one-hot形式进行交叉熵计算。
熵，熵的本质是香农信息量的期望。
熵在信息论中代表随机变量不确定度的度量。一个离散型随机变量X的熵 H(X)定义为：

![upload successful](/images/softmax-3.png)
交叉熵刻画的是实际输出概率和期望输出概率的距离，交叉熵的值越小，则两个概率分布越接近，即实际与期望差距越小。假设概率分布p(xi)为期望输出，概率分布为q(xi)为实际输出，H(X)为交叉熵。则交叉熵的计算表达式为:

![upload successful](/images/softmax-4.png)

### 二、Fashion-MNIST数据集



#### 1、简介

Fashion-MNIST 是一个替代 MNIST 手写数字集 的图像数据集。 它是由 Zalando（一家德国的时尚科技公司）旗下的研究部门提供。其涵盖了来自 10 种类别的共 7 万个不同商品的正面图片。
Fashion-MNIST 的大小、格式和训练集/测试集划分与原始的 MNIST 完全一致。60000/10000 的训练测试数据划分，28x28 的灰度图片。你可以直接用它来测试你的机器学习和深度学习算法性能，且不需要改动任何的代码。

#### 2、数据集的内容

训练集和测试集数据的格式相同，通过获取迭代器的第一个元素可以得知，数据格式是一个具有两个元素的列表。第一个列表表示28*28图像的Tensor表示，第二个元素是一个1个元素的Tensor，它的取值为0-9，分别代表10类图像。



![upload successful](/images/softmax1.png)

![upload successful](/images/softmax.png)

### 三、具体代码实现

#### 1、使用到的包

``

```python
import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")
```

#### 2、加载数据集

通过Pytorch中的`torchvision.datasets`提供的函数进行数据集的加载

``

```python
def load_data_fashion_mnist(batch_size, resize=None):
    """获得数据集，并加载成iterable类型"""
    #定义转换函数trans，使用transforms.ToTensor()将图像转化成Tensor形式
    #class torchvision.transforms.ToTensor
    #把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4),
            data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True, num_workers=4))
```

其中，使用`ToTensor()`方法实际上是一种对图像的归一化处理。假设原图像是8位灰度图像，那么读入的像素矩阵最大值为256，最小值为1，定义矩阵为I，J＝I／256，就是归一化的图像矩阵，就是说归一化之后所有的像素值都在［0，1］区间内。

归一化处理的好处在于：  
(1)归一化能够防止净输入绝对值过大引起的神经元输出饱和现象。  
(2)归一化可以加快训练网络的收敛性.~~（实际上不知道为什么要归一化，待进一步学习回来填坑）~~

`transforms.Compose()`可以将多个转换函数组合在一起，参数是一个由转换函数组成的列表，如trans表示的那样。

`torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)`   
函数功能：使用`torchvision.datasets`提供的数据集加载方法下载并加载MNIST数据集
参数的解析：`root`：存放数据集的目录，`train`：True表示训练集，False表示测试集。 `download` : `True` = 从互联网上下载数据集, `transform`：转换函数，对数据集图像进行一些处理。

`data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)`  
函数功能：


![upload successful](/images/softmax3.png)

#### 3、计算精确度

``

```python
def accuracy(y_hat, y):
    """函数功能：统计预测正确的数量"""
    """网络中输入batch_size*784,网络线性层784*10，结果为batch_size*10的矩阵，第二个维度为预测结果，最大的索引即为预测类别"""
    """numpy 返回最大的元素索引，def argmax(a, axis=None, out=None)
    a—-输入array
    axis—-为0代表列方向，为1代表行方向
    out—-结果写到这个array里面"""
    """y.type:[序号]"""
    y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


class Accumulator:
    """在n个变量上累加。"""
    #新建一个长度为n的列表，列表中每个元素都是一个待累加的变量
    def __init__(self, n):
        self.data = [0.0] * n
    #add方法实现将data的值与add方法参数传入的值进行加和，zip方法实现将iterable对象打包成元组的列表
    """
    >>>a = [1,2,3]
    >>> b = [4,5,6]
    >>> c = [4,5,6,7,8]
    >>> zipped = zip(a,b)     # 打包为元组的列表
    [(1, 4), (2, 5), (3, 6)]
    """
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()#将模型置于评估模式，不计算梯度
    metric = Accumulator(2)
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())#numel()返回数组中元素个数
    #此时metric[0]存放的是网络中预测准确的个数，metric[1]存放数据集中的标签的总数
    return metric[0] / metric[1]
```

#### 4、定义网络结构

使用torch提供的`nn.Sequential`方法创建一个顺序容器，`Modules` 会以他们传入的顺序被添加到容器中。并使用使用torch提供的`nn.init`方法对线性层的权值`weight`初始化为均值为0，标准差为0.01的正态分布。

```python
#Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。实际完成的功能为将28*28的图像矩阵转换成1*784的一维向量。
#FashionMnist数据集为28*28=784的灰度图像，共有10个分类，因此为784*10的线性层
#class torch.nn.Linear(in_features, out_features, bias=True)
#Linear的两个参数：weight和bias
#定义网络结构
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 10)
)

#网络权值初始化
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, 0, 0.01)
net.apply(init_weights)
```
通过阅读代码可以发现在顺序容器nn.Sequential里面并没有定义和Softmax相关的层次，实际上Pytorch实现中将交叉熵损失和softmax结合在了一起，网络采用具有10个输出的线性模型即可。


#### 五、定义优化器与损失函数

使用交叉熵损失函数度量预测概率的准确性。并使用随机梯度下降作为模型的优化器。

```python
#定义损失函数
loss = torch.nn.CrossEntropyLoss()
#定义优化器
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
```

#### 六、定义训练器与训练函数

```python
def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    #统计每一个epoch,损失函数总和、预测正确的数量、样本总数
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)#预测值
        l = loss(y_hat, y)#计算交叉熵损失
        updater.zero_grad()#
        l.backward()
        updater.step()
        metric.add(l, accuracy(y_hat, y), y.numel())
    #返回训练损失和训练准确率
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    if __name__ == '__main__':
        for epoch in range(num_epochs):
            train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
            test_acc = evaluate_accuracy(net, test_iter)
            print(f'epoch:{epoch + 1},训练集损失:{train_metrics[0]},训练集准确率:{train_metrics[1]},测试集准确率:{test_acc}')
```

#### 七、数据加载与调用训练器进行训练

```python
#加载数据
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
num_epochs = 100
#开始训练
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```
#### 八、结果展示

![upload successful](/images/softmax-result.png)

#### 九、总结
如结果图展示，仅仅在线性模型的基础上加上一层Softmax并采用交叉熵损失即可实现分类任务。但是准确率较低只有0.85左右。以后可以尝试更多的Epoch和更深的网络层次查看是否可以达到更强的效果。