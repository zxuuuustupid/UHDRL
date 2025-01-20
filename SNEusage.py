from matplotlib import pyplot as plt
import pandas as pd
import SNE

# 读取十个CSV文件
df1 = pd.read_csv('train_result/output_features/gearbox_anomaly.csv', header=None)
df2 = pd.read_csv('train_result/output_features/gearbox_arch.csv', header=None)
df3 = pd.read_csv('train_result/output_features/gearbox_health.csv', header=None)
d3 = pd.read_csv('train_result/output_features/gearbox_health3.csv', header=None)
d4 = pd.read_csv('train_result/output_features/gearbox_health4.csv', header=None)
d5 = pd.read_csv('train_result/output_features/gearbox_health5.csv', header=None)
d6 = pd.read_csv('train_result/output_features/gearbox_health6.csv', header=None)
d7 = pd.read_csv('train_result/output_features/gearbox_health7.csv', header=None)
d8 = pd.read_csv('train_result/output_features/gearbox_health8.csv', header=None)
d9 = pd.read_csv('train_result/output_features/gearbox_health9.csv', header=None)

# 将DataFrame转换为numpy数组
X1 = df1.values
X2 = df2.values
X3 = df3.values
X4 = d3.values
X5 = d4.values
X6 = d5.values
X7 = d6.values
X8 = d7.values
X9 = d8.values
X10 = d9.values

# 进行t-SNE降维
X_norm1 = SNE.sen_huatu(X1, 30)
X_norm2 = SNE.sen_huatu(X2, 30)
X_norm3 = SNE.sen_huatu(X3, 30)
X_norm4 = SNE.sen_huatu(X4, 30)
X_norm5 = SNE.sen_huatu(X5, 30)
X_norm6 = SNE.sen_huatu(X6, 30)
X_norm7 = SNE.sen_huatu(X7, 30)
X_norm8 = SNE.sen_huatu(X8, 30)
X_norm9 = SNE.sen_huatu(X9, 30)
X_norm10 = SNE.sen_huatu(X10, 30)

# 创建一个空的DataFrame来存储所有降维后的数据
all_data = pd.DataFrame()

# 将每个数据集的降维结果添加到DataFrame中，并添加一个列来区分不同的数据集
all_data = all_data._append(pd.DataFrame({'Dimension 1': X_norm1[:, 0], 'Dimension 2': X_norm1[:, 1], 'Dataset': 'Anomaly'}))
# all_data = all_data._append(pd.DataFrame({'Dimension 1': X_norm2[:, 0], 'Dimension 2': X_norm2[:, 1], 'Dataset': 'Arch'}))
# all_data = all_data._append(pd.DataFrame({'Dimension 1': X_norm3[:, 0], 'Dimension 2': X_norm3[:, 1], 'Dataset': 'Health'}))
# all_data = all_data._append(pd.DataFrame({'Dimension 1': X_norm4[:, 0], 'Dimension 2': X_norm4[:, 1], 'Dataset': 'H3'}))
# all_data = all_data._append(pd.DataFrame({'Dimension 1': X_norm5[:, 0], 'Dimension 2': X_norm5[:, 1], 'Dataset': 'H4'}))
# all_data = all_data._append(pd.DataFrame({'Dimension 1': X_norm6[:, 0], 'Dimension 2': X_norm6[:, 1], 'Dataset': 'H5'}))
# all_data = all_data._append(pd.DataFrame({'Dimension 1': X_norm7[:, 0], 'Dimension 2': X_norm7[:, 1], 'Dataset': 'H6'}))
# all_data = all_data._append(pd.DataFrame({'Dimension 1': X_norm8[:, 0], 'Dimension 2': X_norm8[:, 1], 'Dataset': 'H7'}))
# all_data = all_data._append(pd.DataFrame({'Dimension 1': X_norm9[:, 0], 'Dimension 2': X_norm9[:, 1], 'Dataset': 'H8'}))
# all_data = all_data._append(pd.DataFrame({'Dimension 1': X_norm10[:, 0], 'Dimension 2': X_norm10[:, 1], 'Dataset': 'H9'}))

# 将所有数据保存到一个CSV文件
all_data.to_csv('train_result/output_features/all_norm_data.csv', index=False)

# 创建二维图形
plt.figure(figsize=(10, 6))

# 绘制第一个数据集的散点图（anomaly，红色）
plt.scatter(X_norm1[:, 0], X_norm1[:, 1], c='red', label='Anomaly')

# 绘制其他数据集的散点图，颜色自定义
# plt.scatter(X_norm2[:, 0], X_norm2[:, 1], c='green', label='Arch')
# plt.scatter(X_norm3[:, 0], X_norm3[:, 1], c='blue', label='Health')
# plt.scatter(X_norm4[:, 0], X_norm4[:, 1], c='orange', label='H3')
# plt.scatter(X_norm5[:, 0], X_norm5[:, 1], c='purple', label='H4')
# plt.scatter(X_norm6[:, 0], X_norm6[:, 1], c='brown', label='H5')
# plt.scatter(X_norm7[:, 0], X_norm7[:, 1], c='pink', label='H6')
# plt.scatter(X_norm8[:, 0], X_norm8[:, 1], c='gray', label='H7')
# plt.scatter(X_norm9[:, 0], X_norm9[:, 1], c='olive', label='H8')
# plt.scatter(X_norm10[:, 0], X_norm10[:, 1], c='cyan', label='H9')

# 添加图例
plt.legend()

# 添加标题和坐标轴标签
plt.title('t-SNE 2D Visualization')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')

# 显示图形
plt.show()