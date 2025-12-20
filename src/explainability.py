import os
import sys
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from catboost import CatBoostClassifier

# ================= Setup & Config =================

# 路径处理
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    import src.config as config
except ImportError:
    import config

# 绘图配置
plt.switch_backend('Agg') # 后台绘图模式，适合服务器环境
plt.rcParams.update({
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'],
    'axes.unicode_minus': False
})

# ================= Helper Functions =================

def _load_resources():
    """加载模型和数据"""
    model_path = Path(config.BASE_DIR) / 'outputs' / 'catboost_model.cbm'
    data_path = Path(config.DATA_PROCESSED) / config.PROCESSED_FILE
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found at {data_path}")

    print(f"Loading model from: {model_path.name}")
    model = CatBoostClassifier()
    model.load_model(str(model_path))

    print(f"Loading data from: {data_path.name}")
    df = pd.read_csv(data_path)
    
    return model, df

def _align_features(X, base_dir):
    """
    关键步骤：确保预测数据的特征列与训练时完全一致
    1. 加载训练时的特征列表
    2. 补全缺失列（填0）
    3. 删除多余列
    4. 强制排序
    """
    feature_map_path = Path(base_dir) / 'outputs' / 'model_feature_cols.pkl'
    
    if not feature_map_path.exists():
        print("Warning: Feature map not found. Using raw columns (Risk of mismatch).")
        return X

    try:
        train_cols = joblib.load(feature_map_path)
        print(f"Feature Alignment: Aligning to {len(train_cols)} training features...")
    except Exception as e:
        print(f"Error loading feature map: {e}")
        return X

    # 1. 补全缺失
    missing = set(train_cols) - set(X.columns)
    if missing:
        print(f"  -> Filling {len(missing)} missing columns with 0")
        for c in missing:
            X[c] = 0

    # 2. 剔除多余
    extra = set(X.columns) - set(train_cols)
    if extra:
        print(f"  -> Dropping {len(extra)} extra columns")
        X = X.drop(columns=list(extra))

    # 3. 强制排序
    return X[train_cols]

def _save_plots(shap_values, X_sample, outputs_dir):
    """生成并保存SHAP分析图表"""
    outputs_dir = Path(outputs_dir)
    
    # 1. Summary Plot (Beeswarm)
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    sum_path = outputs_dir / 'shap_summary_dot.png'
    plt.savefig(sum_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {sum_path.name}")

    # 2. Feature Importance Bar Plot
    # 处理多分类 vs 二分类的 shape 问题
    if isinstance(shap_values, list):
        vals = np.abs(shap_values[0]).mean(0)
    else:
        vals = np.abs(shap_values).mean(0)

    imp_df = pd.DataFrame({
        'Feature': X_sample.columns,
        'Importance': vals
    }).sort_values('Importance', ascending=True) # 升序方便画横向条形图

    # 保存 CSV
    csv_path = outputs_dir / 'shap_feature_importance.csv'
    imp_df.sort_values('Importance', ascending=False).to_csv(csv_path, index=False)
    
    # 画图
    plt.figure(figsize=(10, max(6, len(imp_df) * 0.3))) # 动态高度
    bars = plt.barh(imp_df['Feature'], imp_df['Importance'], color=plt.cm.viridis(np.linspace(0.3, 0.9, len(imp_df))))
    
    plt.xlabel('mean(|SHAP value|)')
    plt.title('SHAP Feature Importance')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    # 标数值
    plt.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    
    bar_path = outputs_dir / 'shap_importance_bar.png'
    plt.savefig(bar_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {bar_path.name}")

# ================= Main Execution =================

def explain_model():
    print(f"\n{'='*20} Starting SHAP Analysis {'='*20}")
    
    try:
        # 1. Load Resources
        model, df = _load_resources()
        
        # 2. Preprocessing
        ignored_cols = ['Risk_Label', '姓名', '学号', 'id', 'avg_score']
        # 排除非特征列
        X_raw = df.drop(columns=[c for c in ignored_cols if c in df.columns])
        
        if 'Cluster_Label' in X_raw.columns:
            X_raw['Cluster_Label'] = X_raw['Cluster_Label'].fillna(0).astype(int)

        # 3. Feature Alignment (关键！)
        X = _align_features(X_raw, config.BASE_DIR)
        
        # 4. Compute SHAP
        print("Computing SHAP values (this may take a while)...")
        explainer = shap.TreeExplainer(model)
        
        # 采样加速：如果数据量太大，只取 500 条
        sample_size = min(500, len(X))
        X_sample = X.sample(n=sample_size, random_state=42)
        shap_values = explainer.shap_values(X_sample)
        
        # 5. Visualization
        out_dir = Path(config.BASE_DIR) / 'outputs'
        out_dir.mkdir(parents=True, exist_ok=True)
        
        _save_plots(shap_values, X_sample, out_dir)
        
        print(f"\n{'='*20} Analysis Complete {'='*20}")

    except Exception as e:
        print(f"\n❌ Critical Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    explain_model()