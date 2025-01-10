import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score

seed = 50
# load dataset
wine = load_wine()
x = wine.data
y = wine.target

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=seed)

# train
clf = DecisionTreeClassifier(criterion='entropy', random_state=seed)
clf.fit(x_train, y_train)

# (1)输出决策树图像
plt.figure(figsize=(16, 16))
tree.plot_tree(clf, filled=True, feature_names=wine.feature_names, class_names=wine.target_names)
plt.title("Decision_Tree for Wine Dataset")
plt.savefig("decision_tree.png")  # 保存图像
plt.show()

# (2)计算预测准确度
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"预测准确度: {accuracy:.2f}")
#
# (3)画出决策树层数与预测准确度的关系图
depths = range(1, 7)
accuracies = []

for depth in depths:
    clf_depth = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=seed)
    clf_depth.fit(x_train, y_train)
    y_pred_depth = clf_depth.predict(x_test)
    accuracies.append(accuracy_score(y_test, y_pred_depth))

# 绘制图像
plt.figure(figsize=(16, 8))
plt.plot(depths, accuracies, marker='o')

# 添加每个点的数值
for i, acc in enumerate(accuracies):
    plt.text(depths[i], acc, f'{acc:.2f}', fontsize=12, ha='center', va='bottom')

plt.title("Accuracy vs. Decision_Tree Depth")
plt.xlabel("Decision Tree Depth")
plt.ylabel("Accuracy")
plt.xticks(depths)
plt.grid()
plt.savefig("Accuracy_vs_Depth.png")  # 保存图像
plt.show()
