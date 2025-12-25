# src/model_regression.py
import os
import joblib
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import src.config as config
except ImportError:
    import config

def train_regression_model():
    """
    训练回归模型：预测挂科数目
    输入: SCL-90因子 + 聚类标签
    输出: 预测的挂科数量 (浮点数，实际使用可四舍五入)
    """
    print("\n>>> [Regression Expert] 启动挂科数目预测模型训练...")
    
    # 1. 加载带有聚类标签的数据
    data_path = os.path.join(config.DATA_PROCESSED, 'scl90_with_clusters.csv')
    if not os.path.exists(data_path):
        print(f"❌ 错误: 数据文件 {data_path} 不存在，请先运行 feature_cluster.py")
        return None

    df = pd.read_csv(data_path)
    
    # 2. 特征准备
    # 特征 = SCL-90因子 + 聚类标签 (作为强先验知识)
    feature_cols = config.SCL90_FEATURES + ['Cluster_Label']
    target_col = config.TARGET_REGRESSION
    
    X = df[feature_cols]
    y = df[target_col]
    
    # 指定类别特征的索引 (CatBoost需要知道哪些列是分类变量)
    # Cluster_Label 是最后一列
    cat_features_indices = [len(feature_cols) - 1] 
    
    # 3. 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
    )
    
    # 4. 模型配置与训练
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        loss_function='RMSE',       # 回归任务的标准损失函数
        eval_metric='MAE',          # 评估指标用平均绝对误差，更直观（平均预测偏离几科）
        random_seed=config.RANDOM_SEED,
        verbose=100,
        early_stopping_rounds=50
    )
    
    # 创建专门的数据池，标记类别特征
    train_pool = Pool(X_train, y_train, cat_features=cat_features_indices)
    test_pool = Pool(X_test, y_test, cat_features=cat_features_indices)
    
    print(f"开始训练 (训练集: {len(X_train)}, 测试集: {len(X_test)})...")
    model.fit(train_pool, eval_set=test_pool)
    
    # 5. 模型评估
    preds = model.predict(test_pool)
    
    # 修正预测值：挂科数不能为负数
    preds = np.maximum(preds, 0)
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print("\n--- 回归模型评估报告 ---")
    print(f"RMSE (均方根误差): {rmse:.4f}")
    print(f"MAE  (平均绝对误差): {mae:.4f}  <-- 平均预测偏差约 {mae:.2f} 科")
    print(f"R2 Score: {r2:.4f}")
    
    # 6. 保存模型
    save_path = os.path.join(config.OUTPUT_DIR, 'catboost_regression.cbm')
    model.save_model(save_path)
    print(f"✅ 回归模型已保存至: {save_path}")
    
    return model

if __name__ == "__main__":
    train_regression_model()