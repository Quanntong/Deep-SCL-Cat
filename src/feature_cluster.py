# src/feature_cluster.py
import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    import src.config as config
except ImportError:
    import config

def process_clustering():
    """
    执行聚类特征工程：
    1. 读取清洗后的数据
    2. 对SCL-90特征进行标准化
    3. 执行K-Means聚类 (k=3)
    4. 保存模型和带有聚类标签的新数据
    """
    print(">>> [Cluster Expert] 启动聚类分析...")
    
    # 1. 读取数据
    data_path = os.path.join(config.DATA_PROCESSED, config.PROCESSED_FILE)
    if not os.path.exists(data_path):
        print(f"❌ 错误: 找不到处理后的数据文件 {data_path}，请先运行 data_loader.py")
        return None

    df = pd.read_csv(data_path, encoding='utf-8-sig')
    
    # 2. 提取特征并标准化
    # 只使用SCL-90心理因子进行聚类，不包含人口学特征，保证画像纯粹性
    feature_cols = config.SCL90_FEATURES
    X = df[feature_cols]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. K-Means 聚类
    # 这里保持原论文/设计的 k=3 (通常对应: 健康、亚健康、高风险)
    kmeans = KMeans(n_clusters=3, random_state=config.RANDOM_SEED, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # 4. 结果整合
    df['Cluster_Label'] = cluster_labels
    
    # 简单的画像分析打印
    print("\n--- 聚类结果分析 ---")
    print(f"各簇样本分布:\n{df['Cluster_Label'].value_counts().sort_index()}")
    
    # 计算每个簇在关键因子(如抑郁)上的均值，帮助确认哪个簇是'高危簇'
    # 注意：K-Means的标签0,1,2是随机的，需要人工或后续逻辑确认含义
    print("\n各簇'抑郁'因子均值:")
    print(df.groupby('Cluster_Label')['抑郁'].mean())
    
    # 5. 保存所有资产
    # 保存带有标签的数据，供后续分类和回归模型使用
    save_path = os.path.join(config.DATA_PROCESSED, 'scl90_with_clusters.csv')
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"✅ 已保存带聚类标签的数据至: {save_path}")
    
    # 保存模型对象，用于后续对新样本(单体预测)进行推理
    joblib.dump(kmeans, os.path.join(config.OUTPUT_DIR, 'kmeans.pkl'))
    joblib.dump(scaler, os.path.join(config.OUTPUT_DIR, 'scaler.pkl'))
    print("✅ 已保存 KMeans 模型和 StandardScaler")

    return df

if __name__ == "__main__":
    process_clustering()