import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from pathlib import Path

# 设置绘图风格，处理中文显示问题
# 使用与 config.py 一致的中文字体配置
import matplotlib
import matplotlib.font_manager as fm

# 先设置样式
plt.style.use('seaborn-v0_8-whitegrid')

# 添加中文字体路径到字体管理器
# 获取系统中可用的中文字体
chinese_fonts = []
for font in fm.fontManager.ttflist:
    font_name = font.name.lower()
    if 'yahei' in font_name or 'simhei' in font_name or 'simsun' in font_name or 'microsoft jhenghei' in font_name:
        chinese_fonts.append(font.name)

# 设置字体配置（在样式设置之后，确保覆盖样式中的字体设置）
if chinese_fonts:
    matplotlib.rcParams['font.sans-serif'] = chinese_fonts + ['DejaVu Sans', 'Arial Unicode MS']
else:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans', 'Arial Unicode MS']

plt.rcParams['axes.unicode_minus'] = False

def find_optimal_threshold(model, X, y_true, target_recall=0.80, save_dir=None):
    """
    寻找最佳分类阈值策略：
    在满足 召回率 >= target_recall 的前提下，寻找 精确率(Precision) 最高的点。
    
    参数:
        model: 已训练好的模型 (需支持 predict_proba)
        X: 特征数据 (DataFrame 或 numpy array)
        y_true: 真实标签 (0/1)
        target_recall: 目标召回率 (默认 0.80)
        save_dir: 图片保存路径 (可选)
        
    返回:
        best_threshold (float): 最佳阈值
    """
    print(f"\n{'='*20} 启动阈值寻优策略 {'='*20}")
    
    # 1. 获取预测概率
    try:
        # 获取属于类别 1 (高危) 的概率
        y_scores = model.predict_proba(X)[:, 1]
    except AttributeError:
        print("❌ 错误: 模型没有 predict_proba 方法")
        return 0.5

    # 2. 计算 P-R 曲线数据 
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recalls, precisions)

    # 3. 核心算法：寻找满足 recall >= target 的最优截断点
    # 注意：thresholds 的长度比 recalls/precisions 少 1
    # 我们只看 recalls[:-1]，这样长度就跟 thresholds 对齐了
    valid_indices = np.where(recalls[:-1] >= target_recall)[0]
    
    if len(valid_indices) > 0:
        # 在满足召回率要求的所有点中，找 Precision 最高的那个点的索引
        # valid_indices 是原数组中的下标
        best_idx_in_valid = np.argmax(precisions[valid_indices])
        best_idx = valid_indices[best_idx_in_valid]
        
        status_msg = "✅ 已找到满足目标召回率的最佳阈值"
    else:
        # 如果模型太烂，死活达不到目标召回率（比如要求0.99但模型做不到），则退而求其次
        # 选择 recall 最大的那个点（通常意味着阈值极低）
        print(f"⚠️ 警告: 无法满足 Recall >= {target_recall}，已自动调整为最大可能召回率。")
        best_idx = np.argmax(recalls[:-1])
        status_msg = "⚠️ 妥协阈值 (最大召回优先)"

    # 获取结果
    best_thresh = thresholds[best_idx]
    best_r = recalls[best_idx]
    best_p = precisions[best_idx]
    
    print(f"{status_msg}")
    print(f"   - 目标召回率: {target_recall:.2%}")
    print(f"   - 推荐阈值: {best_thresh:.6f}")
    print(f"   - 预期表现: Recall={best_r:.4f}, Precision={best_p:.4f}")

    # 4. 可视化绘制
    if save_dir:
        _plot_pr_tradeoff(recalls, precisions, thresholds, best_idx, pr_auc, target_recall, save_dir)
        
    return float(best_thresh)

def _plot_pr_tradeoff(recalls, precisions, thresholds, best_idx, pr_auc, target_recall, save_dir):
    """内部辅助函数：绘制专业的 P-R 权衡曲线"""
    plt.figure(figsize=(10, 6))
    
    # 绘制主曲线
    plt.plot(recalls, precisions, label=f'P-R Curve (AUC = {pr_auc:.3f})', 
             color='#1f77b4', linewidth=2, alpha=0.8)
    
    # 填充曲线下面积
    plt.fill_between(recalls, precisions, color='#1f77b4', alpha=0.1)
    
    # 标记最佳点
    best_r = recalls[best_idx]
    best_p = precisions[best_idx]
    best_t = thresholds[best_idx]
    
    plt.scatter(best_r, best_p, s=150, c='#d62728', edgecolors='white', zorder=10, 
                label=f'最佳阈值点\n(T={best_t:.3f}, R={best_r:.2f}, P={best_p:.2f})')
    
    # 绘制目标召回率参考线
    plt.axvline(x=target_recall, color='green', linestyle='--', alpha=0.6, 
                label=f'目标召回率 ({target_recall})')
    
    # 装饰图表
    plt.title('精确率-召回率权衡曲线 (Precision-Recall Trade-off)', fontsize=14, pad=15)
    plt.xlabel('召回率 (Recall) - 查全能力', fontsize=12)
    plt.ylabel('精确率 (Precision) - 查准能力', fontsize=12)
    plt.legend(loc='lower left', frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    
    # 保存
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'optimal_threshold_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 优化图表已保存至: {save_path}")

if __name__ == "__main__":
    # 单元测试代码 (模拟数据运行)
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    
    print(">>> 正在运行单元测试...")
    X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.8, 0.2], random_state=42)
    model = LogisticRegression().fit(X, y)
    
    find_optimal_threshold(model, X, y, target_recall=0.85, save_dir='.')
