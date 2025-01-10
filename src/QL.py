import numpy as np
import matplotlib.pyplot as plt

# 定义参数
mu = 1
sigma = 2
num_samples = 1000000  # 样本数量

# 定义函数 f(x)
def f(x):
    return 2 * x + np.sqrt(np.abs(x)) + 3

# 蒙特卡洛方法步骤
# 从正态分布 N(1, 4) 中生成样本
samples = np.random.normal(mu, sigma, num_samples)

# 计算每个样本的 f(x) 值
f_values = f(samples)

# 计算累积均值，用于展示收敛性
cumulative_mean = np.cumsum(f_values) / np.arange(1, num_samples + 1)
expected_value = np.mean(f_values)

# 绘制累积均值的收敛性图
plt.figure(figsize=(10, 6))
plt.plot(cumulative_mean, color='orange', label="Cumulative Mean")
plt.axhline(y=expected_value, color='red', linestyle='--', label=f"Estimated E[f(X)] ≈ {expected_value:.4f}")
plt.title("Convergence of Monte Carlo Estimation of E[f(X)]")
plt.xlabel("Number of Samples")
plt.ylabel("Cumulative Mean of f(X)")
plt.legend()
plt.grid(True)
plt.show()
