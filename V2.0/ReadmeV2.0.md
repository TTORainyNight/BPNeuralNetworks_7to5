# 一、总述

## 1.1
 这是一个使用Python写的神经网络。比较适合于研究神经网络运行过程，各种参数的意义。

## 1.2
 BP分类神经网络，实现7组数据5组分类，当然，你可以根据自己的需求，修改代码。

## 1.3
 无版权，可以任意传播使用修改等，你说是你自己写的也没啥问题。

## 1.4
  1、改用了tensorflow库，选择了可以处理更大数据规模的多层感知机，提高预测质量和适用场景
  2、改用了标准API接口，实现全参数控制
  3、支持调用GPU进行训练（需要tensflowGPU环境）
  4、添加了环境监测文件

# 二、环境部署

## 2.1
 你需要保证你的电脑上有Python运行环境，我的环境是Python 3.10.12版本。当然，其它版本也是可以的。

## 2.2
 需要依赖库，如果未安装，需要在命令行中，找到Python的pip.exe程序，执行如下的命令：
```python
  pip install numpy
  pip install tensorflow
  pip install matplotlib
```
  有关tensorflowGPU版本的安装，不在此展开，您可以上网搜索。如果有条件，我推荐您使用GPU版本。

## 2.3
 安装完成之后，您可以运行```环境检测.py```若没有错误，且正确输出正比例函数直线，则环境安装正确

# 三、运行代码

## 3.1
 手动将两个.py添加到你的工作环境中。从```API接口实例.py```开始运行。

# 四、代码结构

## 4.1
 BPScore.py神经网络的代码在这里面，具体的参数传递已详细写在注释中。

## 4.2
 参数调整：直接在API接口中调整参数。如果您需要接入您的程序，只需要在接口上做更改。

## 4.3
 ```API接口实例.py```这个文件不是神经网络所必须的，它只是一个实例，详细描述了如何调用这个神经网络。

# 五、代码问题

## 5.1
  在进行数据归一化时，因为数据输入为0-99，直接使用了除法。如果你的数据库不是0-99，那么请使用标准的归一化语句:
```python
   X = np.array([[0],[1]])
   X = (X - np.min(X)) / (np.max(X) - np.min(X))
```