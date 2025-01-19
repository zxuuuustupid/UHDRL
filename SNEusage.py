from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import SNE

# 读取三个CSV文件
df1 = pd.read_csv('train_result/output_features/gearbox_anomaly.csv', header=None)
df2 = pd.read_csv('train_result/output_features/gearbox_arch.csv', header=None)
df3 = pd.read_csv('train_result/output_features/gearbox_health.csv', header=None)

# 将DataFrame转换为numpy数组
X1 = df1.values
X2 = df2.values
X3 = df3.values



# 进行t-SNE降维
X_norm1 = SNE.sen_huatu(X1, 80)
X_norm2 = SNE.sen_huatu(X2, 80)
X_norm3 = SNE.sen_huatu(X3, 80)

# 创建三维图形
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制第一个数据集的散点图
ax.scatter(X_norm1[:, 0], X_norm1[:, 1], X_norm1[:, 2], c='red', label='Data1')

# 绘制第二个数据集的散点图
ax.scatter(X_norm2[:, 0], X_norm2[:, 1], X_norm2[:, 2], c='green', label='Data2')

# 绘制第三个数据集的散点图
ax.scatter(X_norm3[:, 0], X_norm3[:, 1], X_norm3[:, 2], c='blue', label='Data3')

# 添加图例
ax.legend()

# 添加标题和坐标轴标签
ax.set_title('t-SNE 3D Visualization')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')

# 显示图形
plt.show()