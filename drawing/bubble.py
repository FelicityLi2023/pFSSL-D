import matplotlib.pyplot as plt

# 示例数据
# 假设我们有15个客户端和10个类的分布数据
num_clients = 10
num_classes = 10

# 用于模拟每个客户端在每个类中的样本数量（可以自行替换成实际数据）
data_distribution = [
[0, 0, 0, 0, 0, 0, 0, 0, 10, 10],  # Client 0
    [0, 0, 0, 0, 0, 0, 0, 0, 10, 10],  # Client 1
    [0, 0, 0, 0, 0, 0, 10, 10, 0, 0],  # Client 0
    [0, 0, 0, 0, 0, 0, 10, 10, 0, 0],  # Client 1
    [0, 0, 0, 0, 10, 10, 0, 0, 0, 0],  # Client 0
    [0, 0, 0, 0, 10, 10, 0, 0, 0, 0],  # Client 1
    [0, 0, 10, 10, 0, 0, 0, 0, 0, 0],  # Client 0
    [0, 0, 10, 10, 0, 0, 0, 0, 0, 0],  # Client 1
    [10, 10, 0, 0, 0, 0, 0, 0, 0, 0],  # Client 0
    [10, 10, 0, 0, 0, 0, 0, 0, 0, 0],  # Client 1
    # .... 继续为其他客户端填充数据
]

# 类标签
class_labels = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
# 设置全局字体大小
# plt.rcParams.update({'font.size': 14})

# 创建气泡图
# 生成气泡图
plt.figure(figsize=(10, 8))

for client_index in range(num_clients):
    for class_index in range(num_classes):
        size = data_distribution[client_index][class_index]
        if size > 0:  # 只绘制有数据的点
            plt.scatter(client_index, class_index, s=size * 150, color="#C41A32", alpha=1,zorder=50)  # `s`控制气泡大小

# 设置刻度，使得每个 Client Index 都显示，并调整字号
# plt.figure(figsize=(8, 7))

plt.xticks(range(num_clients), range(num_clients), fontsize=20)
plt.yticks(range(num_classes), class_labels, fontsize=20)
plt.xlabel("Client Index", fontsize=20)
plt.ylabel("Class Labels", fontsize=20)
plt.title("Visualize Data Distribution for 10 Clients", fontsize=22)
plt.grid(True)
# 调整布局
plt.tight_layout()
# 保存图像，分辨率为300 DPI
plt.savefig("data_distribution.png", dpi=300)
plt.show()