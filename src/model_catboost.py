import os
import sys
import joblib
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score

try:
    import src.config as config
    import src.strategy_recall as strategy_recall  # <--- [新增] 导入策略模块
except ImportError:
    import config
    # 兼容直接运行的情况
    try:
        import strategy_recall
    except ImportError:
        pass

def train_classification_model():
    """
    训练分类模型：预测是否为高危学生（是否挂科）
    输入: SCL-90因子 + 聚类标签
    输出: 0 (正常) / 1 (高危)
    """
    print("\n>>> [Classification Expert] 启动高危预警模型训练...")

    # 1. 加载数据 (使用聚类后的数据)
    data_path = os.path.join(config.DATA_PROCESSED, 'scl90_with_clusters.csv')
    if not os.path.exists(data_path):
        print(f"❌ 错误: 数据文件 {data_path} 不存在，请先运行 feature_cluster.py")
        return None

    df = pd.read_csv(data_path)

    # 2. 准备特征与目标
    # 特征 = SCL-90因子 + 聚类标签
    feature_cols = config.SCL90_FEATURES + ['Cluster_Label']
    target_col = config.TARGET_CLASSIFICATION  # 'Is_High_Risk'
    
    X = df[feature_cols]
    y = df[target_col]
    
    # 检查目标变量是否只有一类 (防止报错)
    if len(y.unique()) < 2:
        print("❌ 错误：目标变量只有一种类别，无法训练！请检查数据清洗步骤。")
        return None
    
    # 类别特征索引 (Cluster_Label 是最后一列)
    cat_features_indices = [len(feature_cols) - 1]

    # 3. 数据划分
    # 使用 stratify 保证训练集/测试集的高危比例一致
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config.TEST_SIZE, 
            random_state=config.RANDOM_SEED, 
            stratify=y
        )
    except ValueError:
        # 如果样本极少导致无法分层，回退到随机划分
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
        )
    
    # 4. 模型配置
    # auto_class_weights='Balanced': 自动平衡正负样本权重
    # eval_metric='Recall': 我们最在乎召回率（宁可误报，不可漏报）
    model = CatBoostClassifier(
        iterations=800,
        learning_rate=0.05,
        depth=6,
        auto_class_weights='Balanced', 
        loss_function='Logloss',
        eval_metric='Recall',      
        random_seed=config.RANDOM_SEED,
        verbose=100,
        early_stopping_rounds=50
    )
    
    train_pool = Pool(X_train, y_train, cat_features=cat_features_indices)
    test_pool = Pool(X_test, y_test, cat_features=cat_features_indices)
    
    print(f"开始训练 (训练集: {len(X_train)}, 测试集: {len(X_test)})...")
    model.fit(train_pool, eval_set=test_pool)
    
    # 5. 模型评估 (基于默认阈值 0.5)
    print("\n--- [Baseline] 默认阈值(0.5)评估 ---")
    preds = model.predict(test_pool)
    try:
        print(classification_report(y_test, preds, target_names=['正常', '高危']))
    except:
        print(classification_report(y_test, preds))
    
    # ================= [核心新增] 阈值寻优策略 =================
    print("\n>>> [Strategy] 正在寻找最佳决策阈值...")
    try:
        # 设定目标：我们要抓住 80% 的高危学生 (Recall >= 0.80)
        # 策略函数会返回满足该条件下 Precision 最高的阈值
        best_thresh = strategy_recall.find_optimal_threshold(
            model=model,
            X=test_pool,     # 使用测试集评估
            y_true=y_test,
            target_recall=0.80, 
            save_dir=config.OUTPUT_DIR
        )
        
        # 保存最佳阈值到文件，供前端 app.py 读取
        thresh_path = os.path.join(config.OUTPUT_DIR, 'best_threshold.txt')
        with open(thresh_path, 'w') as f:
            f.write(str(best_thresh))
        print(f"💾 最佳阈值已保存至: {thresh_path}")
        
    except Exception as e:
        print(f"⚠️ 阈值优化失败，将使用默认值 0.5。原因: {e}")
        # 兜底保存 0.5
        with open(os.path.join(config.OUTPUT_DIR, 'best_threshold.txt'), 'w') as f:
            f.write("0.5")
    # =========================================================

    # 6. 保存模型
    save_path = os.path.join(config.OUTPUT_DIR, 'catboost_classification.cbm')
    model.save_model(save_path)
    print(f"✅ 分类模型已保存至: {save_path}")
    
    # 保存特征列表，供后续推理使用
    joblib.dump(feature_cols, os.path.join(config.OUTPUT_DIR, 'model_feature_cols.pkl'))
    
    return model

if __name__ == "__main__":
    train_classification_model()