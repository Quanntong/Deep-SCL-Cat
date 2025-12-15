import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import matplotlib
# 使用非交互后端，防止在无GUI环境中报错
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, confusion_matrix
import os
import sys

# 配置中文字体（针对 Windows 环境）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 添加src目录到路径，以便导入config模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import src.config as config
except ImportError:
    # 如果直接导入失败，尝试相对导入
    try:
        from . import config
    except ImportError:
        # 最后尝试直接导入config
        import config


def prepare_data():
    """
    准备数据，逻辑同 model_catboost.py 中的 Step 3
    
    返回:
    tuple: (特征数据X, 目标变量y, 原始DataFrame)
    """
    print("步骤1: 准备数据...")
    
    # 加载处理后的数据
    processed_file_path = os.path.join(config.DATA_PROCESSED, config.PROCESSED_FILE)
    
    try:
        df = pd.read_csv(processed_file_path)
        print(f"数据加载成功，形状: {df.shape}")
    except FileNotFoundError:
        print(f"错误: 文件未找到 - {processed_file_path}")
        print("请先运行 feature_cluster.py 生成聚类数据")
        return None, None, None
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None, None, None
    
    # 确保 Cluster_Label 列转换为整数类型
    if 'Cluster_Label' in df.columns:
        df['Cluster_Label'] = df['Cluster_Label'].astype(int)
        print(f"已将 'Cluster_Label' 列转换为整数类型")
    else:
        print("警告: 数据中未找到 'Cluster_Label' 列")
        # 如果没有 Cluster_Label，创建一个虚拟的类别特征
        df['Cluster_Label'] = np.random.randint(0, 3, size=len(df))
        print("已创建虚拟 'Cluster_Label' 列")
    
    # 检查是否存在目标列 'Risk_Label'
    if 'Risk_Label' not in df.columns:
        print("数据中不存在 'Risk_Label' 列，将基于score列生成目标变量")
        
        # 基于 score 列的和来生成标签
        # 如果所有 score 列的平均值大于阈值，则标记为高危 (1)
        score_columns = [col for col in df.columns if 'score' in col.lower()]
        if score_columns:
            # 计算每个样本的得分平均值
            score_mean = df[score_columns].mean(axis=1)
            # 使用中位数作为阈值
            threshold = score_mean.median()
            df['Risk_Label'] = (score_mean > threshold).astype(int)
            print(f"基于 {len(score_columns)} 个score列生成目标变量，阈值: {threshold:.3f}")
        else:
            # 如果没有score列，随机生成
            np.random.seed(config.RANDOM_SEED)
            df['Risk_Label'] = np.random.randint(0, 2, size=len(df))
            print("随机生成目标变量")
        
        # 显示目标变量分布
        risk_distribution = df['Risk_Label'].value_counts()
        print(f"目标变量分布: 正常(0): {risk_distribution.get(0, 0)} 个, 高危(1): {risk_distribution.get(1, 0)} 个")
    
    # 定义特征和目标
    target_column = 'Risk_Label'
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    print(f"特征数据形状: {X.shape}")
    print(f"目标变量形状: {y.shape}")
    
    return X, y, df


def find_optimal_threshold():
    """
    实现阈值寻优策略
    
    返回:
    float: 最佳阈值
    """
    print("=" * 50)
    print("阈值寻优策略")
    print("=" * 50)
    
    # 1. 加载模型
    print("\n步骤1: 加载模型...")
    model_path = os.path.join(config.BASE_DIR, 'outputs', 'catboost_model.cbm')
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件未找到 - {model_path}")
        print("请先运行 model_catboost.py 训练模型")
        return None
    
    try:
        model = CatBoostClassifier()
        model.load_model(model_path)
        print(f"模型加载成功: {model_path}")
        print(f"模型树数量: {model.tree_count_}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None
    
    # 2. 准备数据
    X, y, df = prepare_data()
    if X is None or y is None:
        return None
    
    # 3. 预测概率
    print("\n步骤2: 预测概率...")
    y_pred_proba = model.predict_proba(X)[:, 1]  # 属于"高危(1)"的概率
    print(f"预测概率完成，形状: {y_pred_proba.shape}")
    print(f"预测概率范围: [{y_pred_proba.min():.3f}, {y_pred_proba.max():.3f}]")
    
    # 4. 计算 Precision-Recall 曲线
    print("\n步骤3: 计算 Precision-Recall 曲线...")
    precision, recall, thresholds = precision_recall_curve(y, y_pred_proba)
    
    # 阈值数组比 precision 和 recall 少一个元素
    # precision_recall_curve 返回的 thresholds 是用于计算 precision 和 recall 的阈值
    print(f"计算得到 {len(thresholds)} 个阈值点")
    
    # 5. 寻找最佳阈值
    print("\n步骤4: 寻找最佳阈值...")
    
    # 目标：Recall >= 0.95 的所有阈值中，Precision 最大的点
    target_recall = 0.95
    best_threshold = None
    best_precision = 0
    best_recall = 0
    
    # 寻找满足 Recall >= 0.95 的阈值
    candidate_indices = []
    for i in range(len(thresholds)):
        if recall[i] >= target_recall:
            candidate_indices.append(i)
    
    if candidate_indices:
        # 在满足条件的阈值中，选择 Precision 最大的
        best_idx = candidate_indices[np.argmax([precision[i] for i in candidate_indices])]
        best_threshold = thresholds[best_idx]
        best_precision = precision[best_idx]
        best_recall = recall[best_idx]
        print(f"找到 Recall >= {target_recall} 的阈值，选择 Precision 最大的点")
    else:
        # 如果无法满足 0.95，则取 Recall 最大时的阈值
        print(f"无法找到 Recall >= {target_recall} 的阈值，选择 Recall 最大的点")
        best_idx = np.argmax(recall[:-1])  # 排除最后一个元素（recall=1, precision=0）
        best_threshold = thresholds[best_idx]
        best_precision = precision[best_idx]
        best_recall = recall[best_idx]
    
    print(f"最佳阈值: {best_threshold:.4f}")
    print(f"对应 Precision: {best_precision:.4f}")
    print(f"对应 Recall: {best_recall:.4f}")
    
    # 6. 可视化
    print("\n步骤5: 可视化...")
    plt.figure(figsize=(10, 8))
    
    # 绘制 P-R 曲线
    plt.plot(recall, precision, 'b-', linewidth=2, label='P-R Curve')
    
    # 标注最佳阈值点
    plt.plot(best_recall, best_precision, 'ro', markersize=10, 
             label=f'最佳阈值点 (Recall={best_recall:.3f}, Precision={best_precision:.3f})')
    
    # 添加阈值文本标注
    plt.annotate(f'阈值={best_threshold:.3f}', 
                 xy=(best_recall, best_precision),
                 xytext=(best_recall + 0.05, best_precision - 0.05),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=12, color='red')
    
    # 绘制目标 Recall 线
    plt.axhline(y=best_precision, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=target_recall, color='gray', linestyle='--', alpha=0.5, 
                label=f'目标 Recall={target_recall}')
    
    # 设置图形属性
    plt.xlabel('Recall (召回率)', fontsize=14)
    plt.ylabel('Precision (精确率)', fontsize=14)
    plt.title('Precision-Recall 曲线与最佳阈值选择', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    
    # 保存图片
    outputs_dir = os.path.join(config.BASE_DIR, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    plot_path = os.path.join(outputs_dir, 'pr_curve.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"P-R 曲线图已保存到: {plot_path}")
    
    # 7. 验证：使用最佳阈值进行硬分类
    print("\n步骤6: 验证最佳阈值...")
    
    # 使用最佳阈值进行硬分类
    y_pred_hard = (y_pred_proba > best_threshold).astype(int)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y, y_pred_hard)
    
    print("\n混淆矩阵 (使用最佳阈值):")
    print(f"              预测正常(0)  预测高危(1)")
    print(f"真实正常(0)      {cm[0, 0]:10d}      {cm[0, 1]:10d}")
    print(f"真实高危(1)      {cm[1, 0]:10d}      {cm[1, 1]:10d}")
    
    # 计算各项指标
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
    
    print(f"\n性能指标 (使用阈值={best_threshold:.4f}):")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision_val:.4f}")
    print(f"召回率 (Recall): {recall_val:.4f}")
    print(f"F1分数: {f1_score:.4f}")
    
    # 8. 返回最佳阈值
    return best_threshold


if __name__ == "__main__":
    # 主程序入口
    print("=" * 50)
    print("阈值寻优策略模块")
    print("=" * 50)
    
    # 执行阈值寻优函数
    optimal_threshold = find_optimal_threshold()
    
    if optimal_threshold is not None:
        print("\n" + "=" * 50)
        print("阈值寻优完成!")
        print("=" * 50)
        print(f"\n最终推荐的阈值: {optimal_threshold:.4f}")
        print(f"使用此阈值进行二分类预测:")
        print(f"  如果样本属于'高危'的概率 > {optimal_threshold:.4f}，则预测为'高危(1)'")
        print(f"  否则预测为'正常(0)'")
        
        # 显示阈值对应的实际意义
        print(f"\n阈值解释:")
        print(f"  - 阈值 {optimal_threshold:.4f} 表示模型对'高危'类别的置信度阈值")
        print(f"  - 较低的阈值会使模型更敏感（召回率更高，但可能有更多误报）")
        print(f"  - 较高的阈值会使模型更保守（精确率更高，但可能漏掉一些正例）")
        
        # 建议的使用方式
        print(f"\n建议使用方式:")
        print(f"  1. 在生产环境中，使用此阈值对模型预测结果进行硬分类")
        print(f"  2. 可以根据业务需求调整阈值（如更关注召回率或精确率）")
        print(f"  3. 定期重新评估和调整阈值以适应数据分布变化")
    else:
        print("\n阈值寻优失败，请检查错误信息")
