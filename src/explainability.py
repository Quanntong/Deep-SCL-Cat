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

# 确保中文字体正确显示 - 使用与 config.py 一致的中文字体配置
import matplotlib
import matplotlib.font_manager as fm

# 添加中文字体路径到字体管理器
# 获取系统中可用的中文字体
chinese_fonts = []
for font in fm.fontManager.ttflist:
    font_name = font.name.lower()
    if 'yahei' in font_name or 'simhei' in font_name or 'simsun' in font_name or 'microsoft jhenghei' in font_name:
        chinese_fonts.append(font.name)

# 设置字体配置
if chinese_fonts:
    matplotlib.rcParams['font.sans-serif'] = chinese_fonts + ['DejaVu Sans', 'Arial Unicode MS']
else:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans', 'Arial Unicode MS']

matplotlib.rcParams['axes.unicode_minus'] = False

# 设置默认字体大小
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
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

def _align_features(X, model):
    """
    关键步骤：确保预测数据的特征列与训练时完全一致
    使用模型的实际特征名称进行对齐
    """
    if not hasattr(model, 'feature_names_') or model.feature_names_ is None:
        print("Warning: Model has no feature names. Using raw columns (Risk of mismatch).")
        return X
    
    train_cols = model.feature_names_
    print(f"Feature Alignment: Aligning to {len(train_cols)} model features...")
    print(f"Model features: {train_cols}")
    print(f"Data features: {list(X.columns)}")

    # 创建特征名称映射（处理缩写差异）
    feature_mapping = {}
    for model_feat in train_cols:
        # 尝试找到匹配的数据列
        matched = False
        for data_feat in X.columns:
            if model_feat in data_feat or data_feat in model_feat:
                feature_mapping[model_feat] = data_feat
                matched = True
                break
        
        # 如果没有找到匹配，使用模型特征名
        if not matched:
            feature_mapping[model_feat] = model_feat
    
    # 1. 重命名列以匹配模型特征名
    X_aligned = X.copy()
    for model_feat, data_feat in feature_mapping.items():
        if model_feat != data_feat and data_feat in X_aligned.columns:
            X_aligned = X_aligned.rename(columns={data_feat: model_feat})
            print(f"  -> Renamed '{data_feat}' to '{model_feat}'")
    
    # 2. 补全缺失列（填0）
    missing = set(train_cols) - set(X_aligned.columns)
    if missing:
        print(f"  -> Filling {len(missing)} missing columns with 0: {list(missing)}")
        for c in missing:
            X_aligned[c] = 0

    # 3. 剔除多余列
    extra = set(X_aligned.columns) - set(train_cols)
    if extra:
        print(f"  -> Dropping {len(extra)} extra columns: {list(extra)}")
        X_aligned = X_aligned.drop(columns=list(extra))

    # 4. 强制排序
    return X_aligned[train_cols]

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
        X = _align_features(X_raw, model)
        
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
