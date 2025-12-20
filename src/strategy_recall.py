import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, confusion_matrix
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import src.config as config
except ImportError:
    try:
        from . import config
    except ImportError:
        import config

def prepare_data():
    print("步骤1: 准备数据...")
    processed_file_path = os.path.join(config.DATA_PROCESSED, config.PROCESSED_FILE)
    
    try:
        df = pd.read_csv(processed_file_path)
        print(f"数据加载成功，形状: {df.shape}")
    except Exception as e:
        print(f"读取文件错误: {e}")
        return None, None, None
    
    # 确保 Cluster_Label 列转换为整数
    if 'Cluster_Label' in df.columns:
        df['Cluster_Label'] = df['Cluster_Label'].fillna(0).astype(int)
    
    # 定义要剔除的列（必须与 model_catboost.py 保持完全一致！）
    # 也就是把“姓名”、“学号”等无关信息删掉
    drop_cols = ['Risk_Label', '姓名', '学号', 'id', 'avg_score']
    
    # 准备特征 X 和 目标 y
    if 'Risk_Label' in df.columns:
        y = df['Risk_Label']
    else:
        print("警告：未找到 Risk_Label，无法计算准确率")
        y = None
        
    # 剔除无关列，生成 X
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    print(f"原始特征数据形状: {X.shape}")
    print(f"原始特征列: {list(X.columns)}")
    
    # 检查是否有模型特征列文件，如果有则按照模型训练时的特征顺序重新排列
    model_feature_path = os.path.join(config.BASE_DIR, 'outputs', 'model_feature_cols.pkl')
    if os.path.exists(model_feature_path):
        try:
            import joblib
            model_feature_cols = joblib.load(model_feature_path)
            print(f"加载模型特征列文件: {model_feature_path}")
            print(f"模型训练时的特征列 ({len(model_feature_cols)}个): {model_feature_cols}")
            
            # 检查特征列是否匹配
            missing_cols = [col for col in model_feature_cols if col not in X.columns]
            extra_cols = [col for col in X.columns if col not in model_feature_cols]
            
            if missing_cols:
                print(f"警告: 以下特征在数据中缺失: {missing_cols}")
                # 尝试用0填充缺失列
                for col in missing_cols:
                    X[col] = 0
                    print(f"  已用0填充缺失列: {col}")
            
            if extra_cols:
                print(f"警告: 以下特征在模型中不存在: {extra_cols}")
                # 删除多余列
                X = X.drop(columns=extra_cols)
                print(f"  已删除多余列: {extra_cols}")
            
            # 按照模型训练时的特征顺序重新排列
            X = X[model_feature_cols]
            print(f"已按照模型训练时的特征顺序重新排列")
            
        except Exception as e:
            print(f"加载模型特征列文件失败: {e}")
            print("将使用原始特征顺序")
    else:
        print(f"警告: 模型特征列文件不存在: {model_feature_path}")
        print("将使用原始特征顺序，可能导致特征顺序不匹配错误")
    
    print(f"最终特征数据形状: {X.shape}")
    print(f"最终特征列: {list(X.columns)}")
    
    return X, y, df

def find_optimal_threshold():
    print("=" * 50)
    print("阈值寻优策略")
    print("=" * 50)
    
    # 加载模型
    model_path = os.path.join(config.BASE_DIR, 'outputs', 'catboost_model.cbm')
    if not os.path.exists(model_path):
        print(f"模型未找到: {model_path}")
        return None
    
    try:
        model = CatBoostClassifier()
        model.load_model(model_path)
        print("模型加载成功")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None
    
    # 准备数据
    X, y, df = prepare_data()
    if X is None or y is None:
        return None
    
    # 预测
    print("\n步骤2: 预测概率...")
    try:
        y_pred_proba = model.predict_proba(X)[:, 1]
    except Exception as e:
        print(f"预测失败，可能是特征列对不上: {e}")
        return None
        
    # 计算 P-R 曲线
    precision, recall, thresholds = precision_recall_curve(y, y_pred_proba)
    
    # 寻找最佳阈值 (Recall >= 0.95)
    target_recall = 0.95
    valid_indices = [i for i, r in enumerate(recall) if r >= target_recall]
    
    if valid_indices:
        # 在满足 Recall >= 0.95 的点中，找 Precision 最高的
        best_idx = valid_indices[np.argmax(precision[valid_indices])]
    else:
        best_idx = np.argmax(recall)
        
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    print(f"最佳阈值: {best_threshold:.4f} (Recall: {recall[best_idx]:.4f})")
    
    # 保存图片
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, label='P-R Curve')
    plt.scatter(recall[best_idx], precision[best_idx], c='red', s=100, label='Optimal')
    plt.title(f'P-R Curve (Thresh={best_threshold:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    
    outputs_dir = os.path.join(config.BASE_DIR, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    plt.savefig(os.path.join(outputs_dir, 'pr_curve.png'))
    print("P-R 曲线已保存")
    
    return best_threshold

if __name__ == "__main__":
    find_optimal_threshold()
