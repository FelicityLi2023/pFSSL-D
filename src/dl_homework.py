import numpy as np

# Sigmoid 函数及其导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# 初始化输入、权重、偏置和目标输出
x1, x2 = 5, 10
W = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]  # w1-w12
b1, b2 = 1, 1  # 偏置
target_o1, target_o2 = 0.01, 0.99  # 目标输出
learning_rate = 0.1

# 前向传播
# 计算隐藏层
net_h1 = x1 * W[0] + x2 * W[1] + b1
net_h2 = x1 * W[2] + x2 * W[3] + b1
net_h3 = x1 * W[4] + x2 * W[5] + b1

h1 = sigmoid(net_h1)
h2 = sigmoid(net_h2)
h3 = sigmoid(net_h3)

# 计算输出层
net_o1 = h1 * W[6] + h2 * W[8] + h3 * W[10] + b2
net_o2 = h1 * W[7] + h2 * W[9] + h3 * W[11] + b2

o1 = sigmoid(net_o1)
o2 = sigmoid(net_o2)

# 计算损失（均方误差）
loss = 0.5 * ((o1 - target_o1) ** 2 + (o2 - target_o2) ** 2)
print(f"Initial loss: {loss}")

# 反向传播
# 计算输出层的梯度
d_loss_o1 = o1 - target_o1
d_loss_o2 = o2 - target_o2

# 输出层梯度 (使用链式法则)
d_o1_net_o1 = sigmoid_derivative(net_o1)
d_o2_net_o2 = sigmoid_derivative(net_o2)

# 梯度从输出层反向传播到隐藏层
# 对于 w7 的梯度
d_net_o1_w7 = h1
grad_w7 = d_loss_o1 * d_o1_net_o1 * d_net_o1_w7

# 对于 w3 的梯度
d_net_h2_w3 = x1
d_net_o1_h2 = W[8]
d_net_o2_h2 = W[9]
grad_h2 = (d_loss_o1 * d_o1_net_o1 * d_net_o1_h2 + d_loss_o2 * d_o2_net_o2 * d_net_o2_h2)
grad_w3 = grad_h2 * sigmoid_derivative(net_h2) * d_net_h2_w3

# 更新权重
W[2] -= learning_rate * grad_w3  # 更新 w3
W[6] -= learning_rate * grad_w7  # 更新 w7

print(f"Updated w3: {W[2]}")
print(f"Updated w7: {W[6]}")
