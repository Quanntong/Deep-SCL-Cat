import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # 服务器后端，防止无GUI环境报错
import matplotlib.pyplot as plt
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.metrics import precision_recall_curve

# ================= Setup =================

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    import src.config as config
except ImportError:
    import config

# 全局绘图风格
plt.style.use('seaborn-v0_8-darkgrid')

# ================= Helper Functions =================

def _load_artifacts():
    """加载数据和模型"""
    data_path = Path(config.DATA_PROCESSED) / config.PROCESSED_FILE
    model_path = Path(config.BASE_DIR) / 'outputs' / 'catboost_model.cbm'
    feat_path = Path(config.BASE_DIR) / 'outputs' / 'model_feature_cols.pkl'

    if not data_path.exists() or not model_path.exists():
        raise FileNotFoundError("Data or Model file missing.")

    print(f"Loading data from: {data_path.name}")
    df = pd.read_csv(data_path)
    
    print(f"Loading model from: {model_path.name}")
    model = CatBoostClassifier()
    model.load_model(str(model_path))

    # 加载特征列表（如果存在）
    train_features = None
    if feat_path.exists():
        train_features = joblib.load(feat_path)
        print(f"Loaded feature map: {len(train_features)} features")

    return df, model, train_features

def _prepare_inference_data(df, train_features=None):
    """
    准备推理数据：
    1. 剔除无关列
    2. 处理类别特征
    3. (关键) 强制对齐特征顺序与训练时一致
    """
    if 'Risk_Label' not in df.columns:
        print("Warning: 'Risk_Label' missing, cannot calculate metrics.")
        return None, None

    # 1. 基础清洗
    ignored = ['Risk_Label', '姓名', '学号', 'id', 'avg_score']
    X = df.drop(columns=[c for c in ignored if c in df.columns])
    y = df['Risk_Label']

    if 'Cluster_Label' in X.columns:
        X['Cluster_Label'] = X['Cluster_Label'].fillna(0).astype(int)

    # 2. 特征对齐 (Feature Alignment)
    if train_features:
        # 补全缺失列
        missing = set(train_features) - set(X.columns)
        if missing:
            print(f"Align: Filling {len(missing)} missing columns with 0")
            for c in missing:
                X[c] = 0
        
        # 剔除多余列
        extra = set(X.columns) - set(train_features)
        if extra:
            print(f"Align: Dropping {len(extra)} extra columns")
            X = X.drop(columns=list(extra))
            
        # 强制重排
        X = X[train_features]
    else:
        print("Warning: No feature map provided. Using raw columns (risk of mismatch).")

    return X, y

def _plot_pr_curve(recall, precision, best_idx, best_thresh, output_dir):
    """绘制 P-R 曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label='P-R Curve', linewidth=2, color='#2878B5')
    
    # 标记最佳点
    best_r = recall[best_idx]
    best_p = precision[best_idx]
    plt.scatter(best_r, best_p, c='#C82423', s=100, zorder=5, 
                label=f'Optimal (Th={best_thresh:.3f})')
    
    # 辅助线
    plt.axvline(best_r, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(best_p, color='gray', linestyle='--', alpha=0.5)

    plt.title('Precision-Recall Curve with Optimal Threshold')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    
    save_path = output_dir / 'pr_curve.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Chart saved to: {save_path}")

# ================= Main Logic =================

def find_optimal_threshold(target_recall=0.95):
    print(f"\n{'='*20} Threshold Optimization Strategy {'='*20}")
    
    try:
        # 1. 加载资源
        df, model, train_features = _load_artifacts()
        
        # 2. 准备数据
        X, y = _prepare_inference_data(df, train_features)
        if X is None: return

        # 3. 模型推理
        print("Running inference...")
        y_proba = model.predict_proba(X)[:, 1]

        # 4. 计算 P-R 曲线 [Image of Precision-Recall curve trade-off]
        precision, recall, thresholds = precision_recall_curve(y, y_proba)

        # 5. 寻找最佳阈值 (NumPy Vectorized Approach)
        # 目标：在 Recall >= 0.95 的前提下，Precision 最大的点
        valid_mask = recall[:-1] >= target_recall # recall 长度比 threshold 多 1
        
        if valid_mask.any():
            # 将不满足条件的 precision 设为 -1，然后找最大值的索引
            # 注意：thresholds 长度比 p/r 短 1，但通常 p/r 的最后一个值是 1.0/0.0，索引对应需要注意
            filtered_precision = np.where(valid_mask, precision[:-1], -1)
            best_idx = np.argmax(filtered_precision)
        else:
            print(f"Warning: No threshold meets Recall >= {target_recall}. Maximizing Recall instead.")
            best_idx = np.argmax(recall[:-1])

        best_threshold = thresholds[best_idx]
        best_r = recall[best_idx]
        best_p = precision[best_idx]

        print(f"\n✅ Optimal Threshold Found: {best_threshold:.4f}")
        print(f"   Metrics at this point: Recall={best_r:.4f}, Precision={best_p:.4f}")

        # 6. 保存结果
        output_dir = Path(config.BASE_DIR) / 'outputs'
        _plot_pr_curve(recall, precision, best_idx, best_threshold, output_dir)
        
        return best_threshold

    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    find_optimal_threshold()