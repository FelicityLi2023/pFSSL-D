import numpy as np
import torch


cosine_matrix = [[ 0.99999994, 0.9997159, 0.48625004, 0.49292493, 0.38781407, 0.3873953,
  -0.10638598, -0.11231025, 0.10642518, 0.12234238],
 [ 0.9997159, 1.0000001, 0.4899171, 0.49651948, 0.39627433, 0.3960651,
  -0.10125187, -0.10730356, 0.1127817, 0.12882169],
 [ 0.48625004, 0.4899171, 1,         0.9992921, 0.55523664, 0.5472433, 0.46794537, 0.46104074, 0.4267626, 0.4395482 ],
 [ 0.49292493, 0.49651948, 0.9992921, 1.0000001, 0.54296994, 0.53495395, 0.4722706, 0.46523523, 0.4013388, 0.41413796],
 [ 0.38781407, 0.39627433, 0.55523664, 0.54296994, 0.99999994, 0.9998108, 0.20953417, 0.20043847, 0.20899455, 0.2268709 ],
 [ 0.3873953, 0.3960651, 0.5472433, 0.53495395, 0.9998108, 0.9999999, 0.2035509, 0.1943786, 0.20877479, 0.22666365],
 [-0.10638598, -0.10125187, 0.46794537, 0.4722706, 0.20953417, 0.2035509, 1.0000001, 0.99989676, 0.08013446, 0.0865553 ],
 [-0.11231025, -0.10730356, 0.46104074, 0.46523523, 0.20043847, 0.1943786, 0.99989676, 1,          0.08250917, 0.08869993],
 [ 0.10642518, 0.1127817, 0.4267626, 0.4013388, 0.20899455, 0.20877479, 0.08013446, 0.08250917, 0.99999994, 0.9997198 ],
 [ 0.12234238, 0.12882169, 0.4395482, 0.41413796, 0.2268709, 0.22666365, 0.0865553, 0.08869993, 0.9997198, 1.0000001 ]]

cosine_matrix = np.array(cosine_matrix)

# 定义 Min-Max 标准化函数
def min_max_normalize(matrix):
    min_val = matrix.min()
    max_val = matrix.max()
    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    return normalized_matrix


# # 执行标准化
normalized_cosine = min_max_normalize(cosine_matrix)
print(normalized_cosine)
#
# # 3. KL 散度矩阵倒数并标准化为相似性
# similarity_kl = 1 / (kl_matrix + 1e-8)  # 倒数转换
# kl_matrix = kl_matrix * 10000
# normalized_similarity_kl = min_max_normalize(similarity_kl)
#
# # 4. 哈曼卡顿距离矩阵倒数并标准化为相似性
# similarity_hamming = 1 / (hamming_matrix + 1e-8)  # 倒数转换
# normalized_similarity_hamming = min_max_normalize(similarity_hamming)
#
# # 打印结果
# print("Normalized Cosine Similarity Matrix:\n", normalized_cosine)
# print("Normalized KL Similarity Matrix:\n", normalized_similarity_kl)
# print("Normalized Hamming Similarity Matrix:\n", normalized_similarity_hamming)
