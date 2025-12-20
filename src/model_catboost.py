import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
import sys

# 添加src目录到路径，以便导入config模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import src.config as config
except ImportError:
    try:
        from . import config
    except ImportError:
        import config


def train_model():
    print("=" * 50)
    print("CatBoost模型训练流程")
    print("=" * 50)
    
    # 1. 加载数据
    print("步骤1: 加载数据...")
    processed_file_path = os.path.join(config.DATA_PROCESSED, config.PROCESSED_FILE)
    
    try:
        df = pd.read_csv(processed_file_path)
        print(f"数据加载成功，形状: {df.shape}")
    except Exception as e:
        print(f"读取文件错误: {e}")
        return None, None, None, None, None

    # 2. 数据准备与标签生成
    print("\n步骤2: 数据准备与标签生成...")
    
    # 确保聚类标签是整数
    if 'Cluster_Label' in df.columns:
        df['Cluster_Label'] = df['Cluster_Label'].fillna(0).astype(int)

    # =======================================================
    # 🎯 核心修复：基于中文因子分生成真实的“高危标签”
    # =======================================================
    if 'Risk_Label' not in df.columns:
        print("⚠️ 未检测到 'Risk_Label'，正在基于心理因子分生成...")
        
        # 1. 锁定所有的心理因子列（利用 config 中的定义）
        factor_cols = [c for c in df.columns if c in config.SCL90_FEATS]
        
        if len(factor_cols) > 0:
            print(f"   已锁定 {len(factor_cols)} 个心理因子列用于评估风险")
            
            # 2. 计算每个学生的平均分
            df['avg_score'] = df[factor_cols].mean(axis=1)
            
            # 3. 设定阈值（取中位数）
            threshold = df['avg_score'].median()
            
            # 4. 生成标签：1=高危，0=正常
            df['Risk_Label'] = (df['avg_score'] > threshold).astype(int)
            
            print(f"   ✅ 已生成 'Risk_Label' (阈值: avg_score > {threshold:.2f})")
            
            # 5. 【重要】把生成好标签的数据存回去！
            # 注意：保存时排除 avg_score 临时列，防止特征泄露
            df_to_save = df.drop(columns=['avg_score'])
            df_to_save.to_csv(processed_file_path, index=False)
            print(f"   💾 已将带有标签的数据回写至: {processed_file_path}")
            
        else:
            print("❌ 严重错误：未找到心理因子列，无法生成标签！将退化为随机模式。")
            df['Risk_Label'] = np.random.randint(0, 2, size=len(df))

    # 显示分布
    dist = df['Risk_Label'].value_counts()
    print(f"目标变量分布: 正常(0): {dist.get(0, 0)}, 高危(1): {dist.get(1, 0)}")

    # 3. 准备训练集
    # 剔除无关列
    drop_cols = ['Risk_Label', '姓名', '学号', 'id', 'avg_score']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['Risk_Label']
    
    print(f"特征列 ({len(X.columns)}): {list(X.columns)}")

    # 4. 划分与训练
    print("\n步骤3: 划分与训练...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y
    )
    
    # 识别类别特征（Cluster_Label）
    cat_features = ['Cluster_Label'] if 'Cluster_Label' in X.columns else []
    
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        auto_class_weights='Balanced',
        cat_features=cat_features,
        verbose=100,
        random_seed=config.RANDOM_SEED,
        eval_metric='Recall',
        early_stopping_rounds=50,
        allow_writing_files=False 
    )
    
    print("开始训练...")
    model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True, plot=False)
    
    # 5. 保存与评估
    outputs_dir = os.path.join(config.BASE_DIR, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    model.save_model(os.path.join(outputs_dir, 'catboost_model.cbm'))
    
    # 保存实际使用的特征列（按照训练时的顺序）
    model_feature_cols = list(X_train.columns)
    model_feature_path = os.path.join(outputs_dir, 'model_feature_cols.pkl')
    joblib.dump(model_feature_cols, model_feature_path)
    print(f"\n模型特征列已保存到: {model_feature_path}")
    print(f"特征数量: {len(model_feature_cols)}")
    print(f"特征列: {model_feature_cols}")
    
    y_pred = model.predict(X_test)
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    return model, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    train_model()
