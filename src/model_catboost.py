import os
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ================= Setup =================

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    import src.config as config
except ImportError:
    import config

# ================= Helper Functions =================

def _ensure_risk_labels(df, file_path):
    """
    检查并生成 Risk_Label。
    如果标签不存在，基于心理因子均值生成（>中位数为高危），并回写文件。
    注意：这是为了保持原代码逻辑的副作用。
    """
    if 'Risk_Label' in df.columns:
        return df

    print("Label Log: 'Risk_Label' not found. Generating based on SCL-90 factors...")
    
    # 1. 锁定因子列
    factor_cols = [c for c in df.columns if c in config.SCL90_FEATS]
    if not factor_cols:
        print("Error: No factor columns found. Assigning random labels (Fallback).")
        df['Risk_Label'] = np.random.randint(0, 2, size=len(df))
        return df

    # 2. 计算阈值与标签
    avg_scores = df[factor_cols].mean(axis=1)
    threshold = avg_scores.median()
    df['Risk_Label'] = (avg_scores > threshold).astype(int)
    
    print(f"Label Log: Generated labels using threshold {threshold:.2f}")
    print(f"Label Log: Distribution - {df['Risk_Label'].value_counts().to_dict()}")

    # 3. 回写文件 (保留原功能的副作用)
    try:
        df.to_csv(file_path, index=False)
        print(f"Data Log: Updated dataset saved to {file_path}")
    except Exception as e:
        print(f"Warning: Failed to save updated labels to disk: {e}")

    return df

def _prepare_features(df):
    """
    准备特征矩阵 X 和目标变量 y
    """
    # 排除非特征列
    ignored_cols = {'Risk_Label', '姓名', '学号', 'id', 'avg_score'}
    target_col = 'Risk_Label'

    # 特征清洗
    X = df.drop(columns=[c for c in ignored_cols if c in df.columns])
    y = df[target_col]

    # 处理类别特征
    cat_features = []
    if 'Cluster_Label' in X.columns:
        X['Cluster_Label'] = X['Cluster_Label'].fillna(0).astype(int)
        cat_features = ['Cluster_Label']

    return X, y, cat_features

def run_training_pipeline():
    print(f"\n{'='*20} CatBoost Training Pipeline {'='*20}")
    
    # 1. 路径准备
    data_path = Path(config.DATA_PROCESSED) / config.PROCESSED_FILE
    output_dir = Path(config.BASE_DIR) / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. 加载数据
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return None
        
    df = pd.read_csv(data_path)
    print(f"Data Loaded: {df.shape}")

    # 3. 标签完整性检查 (副作用操作)
    df = _ensure_risk_labels(df, data_path)

    # 4. 特征工程 
    X, y, cat_features = _prepare_features(df)
    print(f"Features: {len(X.columns)} columns")
    print(f"Categorical Features: {cat_features}")

    # 5. 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_SEED, 
        stratify=y
    )

    # 6. 模型训练
    print("\nTraining CatBoost Classifier...")
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        auto_class_weights='Balanced', # 处理类别不平衡
        cat_features=cat_features,
        verbose=100,
        random_seed=config.RANDOM_SEED,
        eval_metric='Recall',
        early_stopping_rounds=50,
        allow_writing_files=False
    )

    model.fit(
        X_train, y_train, 
        eval_set=(X_test, y_test), 
        use_best_model=True, 
        plot=False
    )

    # 7. 评估与保存
    print("\n--- Evaluation on Test Set ---")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 保存模型
    model_path = output_dir / 'catboost_model.cbm'
    model.save_model(str(model_path))
    print(f"Model saved: {model_path.name}")

    # 保存特征列名 (关键步骤，用于后续推理/SHAP)
    feat_path = output_dir / 'model_feature_cols.pkl'
    joblib.dump(list(X_train.columns), feat_path)
    print(f"Feature map saved: {feat_path.name}")

    return model, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    run_training_pipeline()