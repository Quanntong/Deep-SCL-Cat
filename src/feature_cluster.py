import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys

# 添加src目录到路径，以便导入config和data_loader模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import src.config as config
    import src.data_loader as data_loader
except ImportError:
    # 如果直接导入失败，尝试相对导入
    try:
        from . import config
        from . import data_loader
    except ImportError:
        # 最后尝试直接导入
        import config
        import data_loader


def process_clustering():
    """
    执行K-Means聚类与特征增强
    
    返回:
    pd.DataFrame: 包含聚类标签的增强数据
    """
    print("=" * 50)
    print("开始执行K-Means聚类与特征增强")
    print("=" * 50)
    
    # 1. 加载清洗后的数据
    print("步骤1: 加载数据...")
    df = data_loader.load_and_clean_data()
    
    if df is None:
        print("错误: 数据加载失败，无法进行聚类分析")
        return None
    
    print(f"加载的数据形状: {df.shape}")
    print(f"数据列名: {list(df.columns)}")
    
    # 2. 特征选择
    print("\n步骤2: 特征选择...")
    # 明确选择SCL-90中文特征列
    feature_columns = [c for c in df.columns if c in config.SCL90_FEATS]
    
    # 如果没有找到特征列，回退到所有数值列
    if not feature_columns:
        print("警告: 未找到SCL-90中文特征列，回退到所有数值列")
        exclude_columns = ['Label', 'Target', 'Risk']
        existing_exclude = [col for col in exclude_columns if col in df.columns]
        feature_columns = []
        for col in df.columns:
            if col not in existing_exclude and pd.api.types.is_numeric_dtype(df[col]):
                feature_columns.append(col)
    
    print(f"选择的特征列 ({len(feature_columns)} 个): {feature_columns}")
    
    # 提取特征数据
    X = df[feature_columns].copy()
    print(f"特征数据形状: {X.shape}")
    
    # 3. 数据标准化
    print("\n步骤3: 数据标准化...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("数据标准化完成")
    
    # 4. K-Means聚类
    print("\n步骤4: 执行K-Means聚类...")
    n_clusters = 3  # 根据任务要求设置3个簇
    kmeans = KMeans(n_clusters=n_clusters, random_state=config.RANDOM_SEED, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    print(f"K-Means聚类完成，共分为 {n_clusters} 个簇")
    
    # 5. 特征增强：将聚类结果添加到原始DataFrame
    print("\n步骤5: 特征增强...")
    df_clustered = df.copy()
    df_clustered['Cluster_Label'] = cluster_labels
    print(f"已添加聚类标签列 'Cluster_Label'")
    
    # 6. 保存聚类数据结果
    print("\n步骤6: 保存聚类数据结果...")
    # 确保输出目录存在
    os.makedirs(config.DATA_PROCESSED, exist_ok=True)
    
    # 构建完整保存路径
    save_path = os.path.join(config.DATA_PROCESSED, config.PROCESSED_FILE)
    df_clustered.to_csv(save_path, index=False)
    print(f"聚类数据已保存到: {save_path}")
    print(f"保存的数据形状: {df_clustered.shape}")
    
    # 7. 保存模型和特征信息（模型持久化）
    print("\n步骤7: 保存模型和特征信息...")
    # 确保输出目录存在
    outputs_dir = os.path.join(config.BASE_DIR, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    # 记录聚类使用的特征列名列表
    feature_cols = feature_columns.copy()
    
    # 保存标准化器
    scaler_path = os.path.join(outputs_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"标准化器已保存到: {scaler_path}")
    
    # 保存KMeans模型
    kmeans_path = os.path.join(outputs_dir, 'kmeans.pkl')
    joblib.dump(kmeans, kmeans_path)
    print(f"KMeans模型已保存到: {kmeans_path}")
    
    # 保存特征列名
    feature_cols_path = os.path.join(outputs_dir, 'feature_cols.pkl')
    joblib.dump(feature_cols, feature_cols_path)
    print(f"特征列名已保存到: {feature_cols_path}")
    
    # 8. 返回处理后的DataFrame
    return df_clustered, kmeans, X_scaled, feature_columns


def analyze_clusters(df_clustered, kmeans, X_scaled, feature_columns):
    """
    分析聚类结果
    
    参数:
    df_clustered: 包含聚类标签的DataFrame
    kmeans: 训练好的KMeans模型
    X_scaled: 标准化后的特征数据
    feature_columns: 特征列名列表
    """
    print("\n" + "=" * 50)
    print("聚类结果分析")
    print("=" * 50)
    
    # 1. 打印每个簇的样本数量
    cluster_counts = df_clustered['Cluster_Label'].value_counts().sort_index()
    print("\n1. 每个簇的样本数量:")
    for cluster_id, count in cluster_counts.items():
        print(f"   簇{cluster_id}: {count} 个样本 ({count/len(df_clustered)*100:.1f}%)")
    
    # 2. 打印每个簇的中心点（原始尺度）
    print("\n2. 每个簇的中心点（标准化后的特征空间）:")
    cluster_centers_scaled = kmeans.cluster_centers_
    
    for cluster_id in range(kmeans.n_clusters):
        print(f"\n   簇{cluster_id}中心点:")
        center_values = cluster_centers_scaled[cluster_id]
        
        # 打印每个特征的中心值
        for i, feature in enumerate(feature_columns):
            if i < len(center_values):
                print(f"     {feature}: {center_values[i]:.3f}")
    
    # 3. 分析每个簇的特征画像
    print("\n3. 聚类特征画像分析:")
    
    # 计算每个簇在原始数据上的均值
    for cluster_id in range(kmeans.n_clusters):
        cluster_data = df_clustered[df_clustered['Cluster_Label'] == cluster_id]
        cluster_mean = cluster_data[feature_columns].mean()
        
        print(f"\n   簇{cluster_id}特征均值:")
        # 找出最高和最低的特征值
        sorted_features = cluster_mean.sort_values(ascending=False)
        
        # 打印前3个最高特征
        print(f"     最高特征:")
        for feature in sorted_features.index[:3]:
            print(f"       {feature}: {cluster_mean[feature]:.3f}")
        
        # 打印前3个最低特征
        print(f"     最低特征:")
        for feature in sorted_features.index[-3:]:
            print(f"       {feature}: {cluster_mean[feature]:.3f}")
    
    # 4. 提供简单的画像描述
    print("\n4. 可能的聚类画像描述:")
    print("   注: 以下描述基于特征均值，实际解释需结合领域知识")
    
    # 这里可以根据实际特征进行更具体的描述
    # 例如，如果特征包含SCL-90因子，可以给出心理画像
    if any('score' in col.lower() for col in feature_columns):
        print("   - 簇0: 可能为'高症状型' - 多个SCL-90因子分较高")
        print("   - 簇1: 可能为'健康型' - 多数因子分处于中等或较低水平")
        print("   - 簇2: 可能为'中间型' - 因子分介于健康型和高症状型之间")
    else:
        print("   - 簇0: 特征模式A")
        print("   - 簇1: 特征模式B")
        print("   - 簇2: 特征模式C")


if __name__ == "__main__":
    # 主程序入口
    print("=" * 50)
    print("K-Means聚类与特征增强模块")
    print("=" * 50)
    
    # 执行聚类函数
    result = process_clustering()
    
    if result is not None:
        df_clustered, kmeans, X_scaled, feature_columns = result
        
        # 分析聚类结果
        analyze_clusters(df_clustered, kmeans, X_scaled, feature_columns)
        
        # 显示处理后的数据前几行
        print("\n" + "=" * 50)
        print("处理后的数据前5行:")
        print(df_clustered.head())
        
        print("\n" + "=" * 50)
        print("聚类与特征增强完成!")
        print("=" * 50)
        print("Artifacts (scaler, kmeans, features) saved to outputs/.")
    else:
        print("聚类处理失败，请检查错误信息")
