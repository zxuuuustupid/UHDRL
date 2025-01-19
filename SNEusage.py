from matplotlib import pyplot as plt

import SNE
import pandas as pd

# 读取三个Excel文件
df1 = pd.read_csv('train_result/output_features/gearbox_anomaly.csv', header=None)
df2 = pd.read_csv('train_result/output_features/gearbox_arch.csv', header=None)
df3 = pd.read_csv('train_result/output_features/gearbox_health.csv', header=None)

# 将DataFrame转换为numpy数组
X1 = df1.values
X2 = df2.values
X3 = df3.values

X_norm1 = SNE.sen_huatu(X1,30)
X_norm2 = SNE.sen_huatu(X2,30)
X_norm3 = SNE.sen_huatu(X3,30)

plt.figure(figsize=(10, 6))

# 绘制第一个数据集的散点图
plt.scatter(X_norm1[:, 0], X_norm1[:, 1], c='red', label='Data1')

# 绘制第二个数据集的散点图
plt.scatter(X_norm2[:, 0], X_norm2[:, 1], c='green', label='Data2')

# 绘制第三个数据集的散点图
plt.scatter(X_norm3[:, 0], X_norm3[:, 1], c='blue', label='Data3')

# 添加图例
plt.legend()

# 添加标题和坐标轴标签
plt.title('t-SNE Visualization')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')

# 显示图形
plt.show()