import torch
import torch.nn as nn
import numpy as np

#搭建一个2层的神经网络模型
#每层都是线性层
class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2):
        super(TorchModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)#3*5
        self.layer2 = nn.Linear(hidden_size, hidden_size2)#5*2

    def forward(self, x):
        x = self.layer1(x)
        y_pred = self.layer2(x)
        return y_pred

#自定义模型
class DiyModel:
    def __init__(self, w1, b1, w2, b2):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2

    def forward(self, x):
        hidden = np.dot(x, self.w1.T) + self.b1#2*5
        y_pred = np.dot(hidden, self.w2.T) + self.b2#2*2
        return y_pred


#随机准备一个网络输入
x = np.array([[3.1, 1.3, 1.2],
              [2.1, 1.3, 13]])

#建立torch模型
torch_model = TorchModel(3, 5, 2)

print(torch_model.state_dict())

#打印模型权重，权重为随机初始化
torch_model_w1 = torch_model.state_dict()['layer1.weight'].numpy()#weight可以代表权重，既然是一个线性层，那么就是y= w* x + b
torch_model_b1 = torch_model.state_dict()['layer1.bias'].numpy()
torch_model_w2 = torch_model.state_dict()['layer2.weight'].numpy()
torch_model_b2 = torch_model.state_dict()['layer2.bias'].numpy()
print(torch_model_w1, 'torch w1 权重')
print(torch_model_b1, 'torch b1 偏置')
print('-------------')
print(torch_model_w2, 'torch w2 权重')
print(torch_model_b2, 'torch b2 偏置')
print('-------------')
torch_x = torch.FloatTensor(x)
y_pred = torch_model(torch_x)
print('torch模型预测结果：', y_pred)

#与预测模型进行比较
diy_model = DiyModel(torch_model_w1, torch_model_b1, torch_model_w2, torch_model_b2)
#用自己的模型预测
y_pred_diy = diy_model.forward(x)
print('diy模型预测结果：', y_pred_diy)