import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 14分类的ground truth
ground_truth = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# 14分类的预测结果 (更真实的数据，只有第2个超过0.5)
pred_values = [0.08, 0.72, 0.03, 0.15, 0.04, 0.22, 0.18, 0.01, 0.09, 0.26, 0.11, 0.19, 0.07, 0.05]

# 疾病名称列表
disease_names = ['', 'Cardiomegaly', '', '', '', '', '', '', '', '', '', '', '', '']

# 创建x轴位置
x_pos = np.arange(14)

# 创建子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# 上面的子图 - Ground Truth
# 设置不同的颜色：cardiomegaly为蓝色，其他为橙色
colors_gt = ['orange' if i != 1 else 'steelblue' for i in range(14)]
bars1 = ax1.bar(x_pos, ground_truth, color=colors_gt, alpha=0.7)
# 为值为0的位置添加横线标记
for i, val in enumerate(ground_truth):
    if val == 0.0:
        ax1.plot([i-0.3, i+0.3], [0.02, 0.02], color='black', linewidth=2)
ax1.set_ylim(0, 1.2)
ax1.set_xticks([])  # 不显示任何x轴标签
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.tick_params(axis='y', left=False, labelleft=False)
ax1.tick_params(axis='x', bottom=False)

# 下面的子图 - Predictions
# 设置不同的颜色：cardiomegaly为蓝色，其他为橙色
colors = ['orange' if i != 1 else 'steelblue' for i in range(14)]
bars2 = ax2.bar(x_pos, pred_values, color=colors, alpha=0.7)
ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=2)  # 0.5的虚线
ax2.set_ylim(0, 1.0)
ax2.set_xticks([])  # 不显示任何x轴标签
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.tick_params(axis='y', left=False, labelleft=False)
ax2.tick_params(axis='x', bottom=False)

# 调整子图间距
plt.tight_layout()
plt.subplots_adjust(hspace=0.3)

# 保存图片
plt.savefig('medical_classification_plot.png', dpi=300, bbox_inches='tight')
plt.show() 