import numpy as np

# 原始矩阵
matrix = np.array([[1., 0.74027228, 0.6974023, 0.69633138, 0.7080586, 0.71366203, 0.70876676, 0.70875072, 0.69700038, 0.7032491, ],
 [0.74027228, 1., 0.69275022, 0.69341731, 0.71017635, 0.71415913, 0.70519578, 0.70574045, 0.69517601, 0.70292586],
 [0.6974023, 0.69275022, 1., 0.73601395, 0.69692338, 0.69628662, 0.69640446, 0.69388103, 0.71560848, 0.71115351],
 [0.69633138, 0.69341731, 0.73601395, 1., 0.69767034, 0.70192546, 0.69848269, 0.70123422, 0.70498437, 0.70106107],
 [0.7080586, 0.71017635, 0.69692338, 0.69767034, 1., 0.73867935, 0.69562113, 0.69486916, 0.70965958, 0.71564865],
 [0.71366203, 0.71415913, 0.69628662, 0.70192546, 0.73867935, 1., 0.69880807, 0.70123553, 0.70732212, 0.71362633],
 [0.70876676, 0.70519578, 0.69640446, 0.69848269, 0.69562113, 0.69880807, 1., 0.75351191, 0.68702841, 0.6914767, ],
 [0.70875072, 0.70574045, 0.69388103, 0.70123422, 0.69486916, 0.70123553, 0.75351191, 1., 0.68720764, 0.69214088],
 [0.69700038, 0.69517601, 0.71560848, 0.70498437, 0.70965958, 0.70732212, 0.68702841, 0.68720764, 1., 0.78824985],
 [0.7032491, 0.70292586, 0.71115351, 0.70106107, 0.71564865, 0.71362633, 0.6914767, 0.69214088, 0.78824985, 1., ]])
# 找到小于 0.8 的最小值
min_val = matrix[matrix < 0.8].min()

# 定义缩放函数
def scale_values(value, min_val, threshold=0.8, target_low=0.55, target_range=0.05):
    if value < threshold:
        return target_low + (value - min_val) * (target_range / (threshold - min_val))
    return value


# 对矩阵进行缩放
scaled_matrix = np.vectorize(scale_values)(matrix, min_val)

# 格式化输出，保留逗号
formatted_matrix = np.array2string(scaled_matrix, separator=', ', formatter={'float_kind': lambda x: f"{x:.8f}"})

# 输出结果
print(formatted_matrix)
