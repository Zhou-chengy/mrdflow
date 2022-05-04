# Mrdflow 1.1.0 beta
## Based by numpy

## 基于numpy构建

## 下载方式
```
pip install mrdflow
```

## 使用说明：
### 1 autograd
#### 1.1 Tensor数组
##### Tensor是autograd的核心，你可以通过以下方式创建Tensor
```Python
import mrdflow.autograd as ag
x = ag.arange(12)
#创建一个shape为(12,)的Tensor数组x,其功能等同于numpy.arange.
y = ag.zeros(12,12,grad=True)
#创建一个shape为(12,12)的Tensor数组y,其功能等同于numpy.zeros,你可以将grad设置为True，这样可以自动求导
z = ag.randn(12,12,grad=True)
#创建一个shape为(12,12)的Tensor数组z,其功能等同于numpy.random.randn,你可以将grad设置为True，这样可以自动求导
```
##### 可以使用Tensor.gradient求导
``` Python
import mrdflow.autograd as ag
x = ag.arange(12)
y = ag.sin(x/2)
y.gradient()
print(x.grad)
#求导出y对x导数
```
##### Tensor内置了许多函数，以下是个例子
``` Python
import mrdflow.autograd as ag
import numpy as np
x = ag.arange(12).reshape(3,4)
y = ag.arange(12).reshape(4,3)
print(ag.dot(x,y))
#ag.dot：矩阵乘法函数
c = x.F
c = x.T
#Tensor.F:归一化，等同于numpy.ndarray.Flatten()
#Tensor.T:矩阵转置，等同于numpy.transpose(x)
v = x.numpy()
#将x转换成numpy.ndarray
```
##### Tensor数组无法直接转换成numpy数组，必须通过Tensor.numpy()进行转换

#### 1.2 Op算子
##### Tensor数组的运算是基于Op算子的，无论是Exp还是矩阵乘法。Op算子有2个属性,分别是compute和gradient。compute处理计算，gradient进行反向求导。
``` Python
import mrdflow.autograd as ag
class TestOp(ag.Op):
    def compute(inputs:list):
        """进行运算操作,将您的计算结果保存为self.re"""
        return Tensor(self.re,op=self,grad=True)
    def gradient(self,inputs,grad):
        inputs[0].backward(grad)
        #grad*导数值
```
### 2 神经网络
#### 2.1 mnist
##### 下面是用mrdflow训练模型识别手写数字的例子。请确保下载好mnist.npz文件，[下载链接]（https://www.kaggle.com/datasets/vikramtiwari/mnist-numpy/download)
``` Python
import mrdflow as mf
from mrdflow import autograd as ag
import numpy as np
#import cProfile
data = np.load('mnist.npz')
x_train = data['x_train']
y_train = data['y_train']
def x_train_data(x):
    return (ag.Tensor(x)/255).reshape(1,28,28)
def one_hot(y):
    v = ag.zeros(10)
    v[y] = 1
    return v
def test(model,x,y):
    p = len(x)
    o = 0
    for i in range(0,p):
        tx = np.argmax(model.predict(ag.Tensor(x[i]/255)).array)
        ty = y[i]
        if tx==ty:
            o += 1
    return o/p
x_train = list(map(x_train_data,x_train))
y_train = list(map(one_hot,y_train))
model = mf.nn.Sequential([mf.nn.layer.Conv2d([28,28],[4,4],activation=mf.relu,pad='VALID'),
                       mf.nn.layer.MaxPooling2d([5,5]),
                       mf.nn.layer.Dense(5*5,10,activation=mf.softmax)])
model.compile(optimizer=mf.nn.Adam,lr=0.1)
model.fit(x=x_train,y=y_train,epoch=1000,batch_size=10)
#保存模型
```
