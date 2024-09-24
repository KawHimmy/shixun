import pandas as pd
import matplotlib.pyplot as plt

# 读取两个CSV文件
file1 = 'my_model/vgg16/pr_curve_data_20240924_normal.csv'
file2 = 'my_model/vgg16/pr_curve_data_20240924_enhanced.csv'

# 加载CSV文件
data1 = pd.read_csv(file1)
data2 = pd.read_csv(file2)

# 提取precision和recall
precision1, recall1 = data1['Precision'], data1['Recall']
precision2, recall2 = data2['Precision'], data2['Recall']

# 绘制PR曲线
plt.figure(figsize=(8, 6))
plt.plot(recall1, precision1, label='PR Curve 1 normal', color='blue', lw=2)
plt.plot(recall2, precision2, label='PR Curve 2 enhanced', color='red', lw=2)

# 设置图形标题和标签
plt.title('Precision-Recall Curves')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='best')

# 保存图像
plt.savefig('my_model/vgg16/multiple_pr_curves.jpg')
plt.show()
