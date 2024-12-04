import torch
import torch.nn as nn
import numpy as np

#使用torch计算交叉熵
ce_loss = nn.CrossEntropyLoss()
#假设有3个样本，每个样本都在做3分类
pred = torch.FloatTensor([[0.3, 0.1, 0.3],
                          [0.9, 0.2, 0.9],
                          [0.5, 0.4, 0.2]])
#正确的类别分别为1，2，0
target = torch.LongTensor([1, 2, 0])#[0,1,0][0,0,1][1,0,0]

loss = ce_loss(pred, target)
print(loss, 'torch输出交叉熵')

#实现softmax函数，运用softmax函数是因为数组里面的数字相加并不等于1,相当于对样本进行初始化，激活函数，将线性的转化为非线性的
def softmax(matrix):
    return np.exp(matrix) / np.sum(np.exp(matrix), axis = 1, keepdims=True)

#将输入转化为one-hot矩阵，对给出的正确的类别进行初始化
def to_one_hot(target, shape):
    one_hot_target = np.zeros(shape)
    for i, t in enumerate(target):#enumerate(target)会遍历target数组的每一个元素t，
                                  # 同时返回当前元素的索引i，如果 target = [0, 2, 1, 3]，则 enumerate(target) 会依次生成 (0, 0), (1, 2), (2, 1), (3, 3)。
        one_hot_target[i][t] = 1#对于每个索引 i，在 one_hot_target 数组的第 i 行，将第 t 列的值设置为 1。
    return one_hot_target

#手动实现交叉熵，将两个初始化得到的矩阵通过交叉熵进行计算
def cross_entropy(pred, target):
    batch_size, class_num = pred.shape
    pred = softmax(pred)
    target = to_one_hot(target, pred.shape)
    entropy = - np.sum(target * np.log(pred), axis = 1)
    return sum(entropy) / batch_size

print(cross_entropy(pred.numpy(), target.numpy()), '手动实现交叉熵')


