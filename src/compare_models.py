import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score

# ================= Configuration & Setup =================

# 路径处理：优先保证项目根目录在 path 中
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    import src.config as config
except ImportError:
    import config

# 全局绘图设置
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans'],
    'axes.unicode_minus': False
})

warnings.filterwarnings('ignore')

# ================= Helper Functions =================

def _get_baseline_models():
    """定义并返回基线模型字典"""
    models = {}
    seed = config.RANDOM_SEED

    # 1. Random Forest (需处理 NaN)
    models["Random Forest"] = make_pipeline(
        SimpleImputer(strategy='median'),
        RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    )

    # 2. Logistic Regression (需处理 NaN + 标准化)
    models["Logistic Regression"] = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed, n_jobs=-1)
    )

    # 3. SVM (需处理 NaN + 标准化)
    models["SVM"] = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(),
        SVC(probability=True, random_state=seed)
    )

    # 4. XGBoost (可选依赖)
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            random_state=seed, use_label_encoder=False, 
            eval_metric='logloss', n_jobs=-1
        )
    except ImportError:
        print("Build Log: XGBoost not installed, skipping.")

    # 5. LightGBM (可选依赖)
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            random_state=seed, n_jobs=-1, verbose=-1
        )
    except ImportError:
        print("Build Log: LightGBM not installed, skipping.")

    return models

def _calculate_metrics(y_true, y_pred, y_proba=None):
    """计算单个模型的各项指标"""
    metrics = {
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'F1': f1_score(y_true, y_pred, average='weighted'),
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC': np.nan
    }
    
    if y_proba is not None:
        try:
            metrics['AUC'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            pass  # 多分类或数据问题导致无法计算 AUC 时保持 nan
            
    return metrics

def _evaluate_model(model, X_test, y_test, name):
    """执行预测并返回评估结果字典"""
    try:
        y_pred = model.predict(X_test)
        
        # 获取概率：优先尝试 predict_proba，其次 decision_function
        y_proba = None
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_proba = model.decision_function(X_test)

        metrics = _calculate_metrics(y_test, y_pred, y_proba)
        metrics['Model'] = name
        
        # 简单打印
        print(f"[{name:<20}] Recall: {metrics['Recall']:.4f} | F1: {metrics['F1']:.4f}")
        return metrics

    except Exception as e:
        print(f"Error evaluating {name}: {e}")
        return {'Model': name, 'Recall': np.nan, 'Precision': np.nan, 
                'F1': np.nan, 'Accuracy': np.nan, 'AUC': np.nan}

def _plot_results(df, output_dir):
    """生成并保存对比图表"""
    metrics_map = {
        'Recall': '召回率 (Recall)',
        'F1': 'F1分数', 
        'Accuracy': '准确率 (Accuracy)', 
        'AUC': 'AUC'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('多模型性能对比分析', fontsize=16, fontweight='bold')
    
    for idx, (metric, title) in enumerate(metrics_map.items()):
        ax = axes[idx // 2, idx % 2]
        # 过滤 NaN 并排序
        subset = df.dropna(subset=[metric]).sort_values(metric)
        
        if subset.empty:
            continue

        # 颜色逻辑：高亮我们的模型
        colors = ['red' if 'Deep-SCL-Cat' in m else plt.cm.Set3(i/len(subset)) 
                  for i, m in enumerate(subset['Model'])]
        
        bars = ax.barh(subset['Model'], subset[metric], color=colors, edgecolor='grey', alpha=0.8)
        
        # 在柱状图旁标注数值
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
        
        ax.set_xlabel(title)
        ax.set_xlim(0, min(1.0, subset[metric].max() * 1.15))
        
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved to: {save_path}")

# ================= Main Execution =================

def run_comparison(X_train, y_train, X_test, y_test, our_model):
    print(f"\n{'='*20} Starting Model Comparison {'='*20}")
    
    outputs_dir = os.path.join(config.BASE_DIR, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    results = []

    # 1. 获取并训练基线模型
    baselines = _get_baseline_models()
    print(f"Initialized {len(baselines)} baseline models.")
    
    for name, model in baselines.items():
        try:
            model.fit(X_train, y_train)
            res = _evaluate_model(model, X_test, y_test, name)
            results.append(res)
        except Exception as e:
            print(f"Failed to train {name}: {e}")

    # 2. 评估我们的模型 (假设外部已经训练好，或者在这里直接预测)
    # 注意：如果 our_model 还没 fit，这里会报错，遵循原代码逻辑假设已 fit
    print("-" * 60)
    results.append(_evaluate_model(our_model, X_test, y_test, "Deep-SCL-Cat (Ours)"))

    # 3. 整理结果
    df_results = pd.DataFrame(results)
    df_sorted = df_results.sort_values('Recall', ascending=False).reset_index(drop=True)
    
    # 4. 保存与可视化
    csv_path = os.path.join(outputs_dir, 'model_comparison.csv')
    df_sorted.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\nData saved to: {csv_path}")
    
    try:
        _plot_results(df_sorted, outputs_dir)
    except Exception as e:
        print(f"Visualization failed: {e}")

    print(f"\n{'='*20} Summary {'='*20}")
    print(df_sorted.to_string(index=False))
    
    return df_sorted

if __name__ == "__main__":
    print("Comparison module loaded.")