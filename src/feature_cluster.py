import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ================= Setup =================

# 路径适配
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    import src.config as config
    from src import data_loader
except ImportError:
    import config
    import data_loader

# ================= Core Logic =================

def train_cluster_model(df, n_clusters=3):
    """
    执行特征提取、标准化与 K-Means 聚类
    """
    # 1. 特征选择 (优先 SCL90，回退到数值列)
    feature_cols = [c for c in df.columns if c in config.SCL90_FEATS]
    if not feature_cols:
        print("Warning: SCL-90 features not found. Falling back to numeric columns.")
        # 排除非特征列
        exclude = {'Label', 'Target', 'Risk', 'Risk_Label', 'id'}
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                        if c not in exclude]

    print(f"Features selected ({len(feature_cols)}): {feature_cols}")

    # 2. 标准化
    X = df[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. 聚类 
    print(f"Training K-Means with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=config.RANDOM_SEED, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # 4. 合并结果
    df_result = df.copy()
    df_result['Cluster_Label'] = labels
    
    return {
        'df': df_result,
        'model': kmeans,
        'scaler': scaler,
        'features': feature_cols,
        'X_scaled': X_scaled
    }

def save_artifacts(results_dict):
    """
    持久化保存：数据 csv + 模型 pkl
    """
    output_dir = Path(config.BASE_DIR) / 'outputs'
    data_dir = Path(config.DATA_PROCESSED)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 保存数据
    csv_path = data_dir / config.PROCESSED_FILE
    results_dict['df'].to_csv(csv_path, index=False)
    print(f"Data saved: {csv_path.name}")

    # 保存模型组件
    artifacts = {
        'scaler.pkl': results_dict['scaler'],
        'kmeans.pkl': results_dict['model'],
        'feature_cols.pkl': results_dict['features']
    }

    for filename, obj in artifacts.items():
        path = output_dir / filename
        joblib.dump(obj, path)
        print(f"Artifact saved: {filename}")

def analyze_profiles(df, feature_cols):
    """
    利用 Pandas GroupBy 快速生成画像分析
    """
    print(f"\n{'='*20} Cluster Profile Analysis {'='*20}")
    
    # 1. 样本分布
    counts = df['Cluster_Label'].value_counts().sort_index()
    total = len(df)
    print("\n--- Sample Distribution ---")
    for lbl, count in counts.items():
        print(f"Cluster {lbl}: {count} ({count/total:.1%})")

    # 2. 聚类中心 (Feature Means)
    # 使用 groupby 一次性计算所有均值，替代手动循环
    print("\n--- Feature Profiles (Mean Values) ---")
    means = df.groupby('Cluster_Label')[feature_cols].mean()
    
    for lbl in means.index:
        row = means.loc[lbl].sort_values(ascending=False)
        top_3 = row.head(3).index.tolist()
        bottom_3 = row.tail(3).index.tolist()
        
        print(f"Cluster {lbl}:")
        print(f"  Top Features:    {', '.join([f'{f}({row[f]:.2f})' for f in top_3])}")
        print(f"  Lowest Features: {', '.join([f'{f}({row[f]:.2f})' for f in bottom_3])}")

    # 3. 业务解释 (保持原有逻辑)
    print("\n--- Interpretation Hint ---")
    if any('score' in f for f in feature_cols) or any(f in config.SCL90_FEATS for f in feature_cols):
        print("Cluster 0/1/2 interpretation depends on severity (check Top Features).")
        print("Typically: High scores -> Symptomatic; Low scores -> Healthy.")
    else:
        print("Check feature patterns above to assign labels.")

# ================= Main Execution =================

if __name__ == "__main__":
    print("Starting Clustering Pipeline...")
    
    # 1. Load
    raw_df = data_loader.load_and_clean_data()
    
    if raw_df is not None:
        # 2. Train
        results = train_cluster_model(raw_df, n_clusters=3)
        
        # 3. Save
        save_artifacts(results)
        
        # 4. Analyze
        analyze_profiles(results['df'], results['features'])
        
        print("\nPipeline completed successfully.")
    else:
        print("Pipeline aborted due to data loading failure.")