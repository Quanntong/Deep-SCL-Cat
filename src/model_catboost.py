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
    # 如果直接导入失败，尝试相对导入
    try:
        from . import config
    except ImportError:
        # 最后尝试直接导入config
        import config


def train_model():
    """
    构建CatBoost训练流程
    
    返回:
    tuple: (模型, 测试集特征, 测试集标签)
    """
    print("=" * 50)
    print("CatBoost模型训练流程")
    print("=" * 50)
    
    # 1. 加载处理后的数据
    print("步骤1: 加载数据...")
    processed_file_path = os.path.join(config.DATA_PROCESSED, config.PROCESSED_FILE)
    
    try:
        df = pd.read_csv(processed_file_path)
        print(f"数据加载成功，形状: {df.shape}")
        print(f"数据列名: {list(df.columns)}")
    except FileNotFoundError:
        print(f"错误: 文件未找到 - {processed_file_path}")
        print("请先运行 feature_cluster.py 生成聚类数据")
        return None, None, None
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None, None, None
    
    # 2. 数据准备
    print("\n步骤2: 数据准备...")
    
    # 确保 Cluster_Label 列转换为整数类型
    if 'Cluster_Label' in df.columns:
        df['Cluster_Label'] = df['Cluster_Label'].astype(int)
        print(f"已将 'Cluster_Label' 列转换为整数类型")
    else:
        print("警告: 数据中未找到 'Cluster_Label' 列")
        # 如果没有 Cluster_Label，创建一个虚拟的类别特征
        df['Cluster_Label'] = np.random.randint(0, 3, size=len(df))
        print("已创建虚拟 'Cluster_Label' 列")
    
    # 检查是否存在目标列 'Risk_Label'
    if 'Risk_Label' not in df.columns:
        print("数据中不存在 'Risk_Label' 列，将随机生成二分类目标变量用于演示")
        print("注意: 在实际应用中，请确保数据包含真实的目标变量")
        
        # 随机生成二分类目标变量，模拟高危/正常标签
        # 这里使用简单的规则：基于 score 列的和来生成标签
        # 如果所有 score 列的平均值大于阈值，则标记为高危 (1)
        score_columns = [col for col in df.columns if 'score' in col.lower()]
        if score_columns:
            # 计算每个样本的得分平均值
            score_mean = df[score_columns].mean(axis=1)
            # 使用中位数作为阈值
            threshold = score_mean.median()
            df['Risk_Label'] = (score_mean > threshold).astype(int)
            print(f"基于 {len(score_columns)} 个score列生成目标变量，阈值: {threshold:.3f}")
        else:
            # 如果没有score列，随机生成
            np.random.seed(config.RANDOM_SEED)
            df['Risk_Label'] = np.random.randint(0, 2, size=len(df))
            print("随机生成目标变量")
        
        # 显示目标变量分布
        risk_distribution = df['Risk_Label'].value_counts()
        print(f"目标变量分布: 正常(0): {risk_distribution.get(0, 0)} 个, 高危(1): {risk_distribution.get(1, 0)} 个")
    
    # 定义特征和目标
    target_column = 'Risk_Label'
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    print(f"特征数据形状: {X.shape}")
    print(f"目标变量形状: {y.shape}")
    print(f"特征列: {list(X.columns)}")
    
    # 3. 数据集划分
    print("\n步骤3: 数据集划分...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_SEED,
        stratify=y  # 分层抽样，保持类别比例
    )
    
    print(f"训练集形状: {X_train.shape}")
    print(f"测试集形状: {X_test.shape}")
    print(f"训练集类别分布: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"测试集类别分布: {pd.Series(y_test).value_counts().to_dict()}")
    
    # 4. 模型配置
    print("\n步骤4: 模型配置...")
    
    # 识别类别特征
    categorical_features = []
    if 'Cluster_Label' in X.columns:
        categorical_features = ['Cluster_Label']
        print(f"类别特征: {categorical_features}")
    else:
        print("警告: 未找到 'Cluster_Label' 列作为类别特征")
    
    # 初始化CatBoost分类器
    model = CatBoostClassifier(
        iterations=500,           # 迭代次数
        learning_rate=0.05,       # 学习率
        depth=6,                  # 树深度
        auto_class_weights='Balanced',  # 自动处理样本不平衡
        cat_features=categorical_features,  # 类别特征
        verbose=100,              # 训练过程输出频率
        random_seed=config.RANDOM_SEED,  # 随机种子
        eval_metric='Recall',     # 评估指标（重点关注Recall）
        early_stopping_rounds=50, # 早停轮数
        task_type='CPU'           # 使用CPU训练
    )
    
    print("CatBoost模型配置完成")
    print(f"关键参数: iterations=500, learning_rate=0.05, depth=6, auto_class_weights='Balanced'")
    
    # 5. 模型训练
    print("\n步骤5: 模型训练...")
    print("开始训练CatBoost模型（详细日志每100次迭代输出一次）...")
    
    try:
        model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            use_best_model=True,
            plot=False  # 不显示训练过程图
        )
        print("模型训练完成!")
    except Exception as e:
        print(f"模型训练失败: {e}")
        return None, X_test, y_test
    
    # 6. 模型保存
    print("\n步骤6: 模型保存...")
    
    # 确保输出目录存在
    outputs_dir = os.path.join(config.BASE_DIR, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    # 保存模型
    model_path = os.path.join(outputs_dir, 'catboost_model.cbm')
    model.save_model(model_path)
    print(f"模型已保存到: {model_path}")
    
    # 同时使用joblib保存（可选）
    joblib_path = os.path.join(outputs_dir, 'catboost_model.joblib')
    joblib.dump(model, joblib_path)
    print(f"模型已保存到: {joblib_path}")
    
    # 7. 模型评估
    print("\n步骤7: 模型评估...")
    
    # 在测试集上预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # 正类的概率
    
    # 打印分类报告
    print("\n分类报告 (测试集):")
    report = classification_report(y_test, y_pred, target_names=['正常(0)', '高危(1)'])
    print(report)
    
    # 重点关注Recall指标
    from sklearn.metrics import recall_score
    recall = recall_score(y_test, y_pred, average='weighted')
    print(f"加权平均Recall: {recall:.4f}")
    
    # 显示特征重要性
    print("\n特征重要性 (前10个):")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.get_feature_importance()
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    print(feature_importance.head(10).to_string(index=False))
    
    # 8. 返回结果
    return model, X_test, y_test


if __name__ == "__main__":
    # 主程序入口
    print("=" * 50)
    print("CatBoost分类模型训练")
    print("=" * 50)
    
    # 执行训练函数
    model, X_test, y_test = train_model()
    
    if model is not None:
        print("\n" + "=" * 50)
        print("模型训练流程完成!")
        print("=" * 50)
        
        # 显示模型基本信息
        print(f"\n模型树数量: {model.tree_count_}")
        print(f"模型参数: {model.get_params()}")
        
        # 如果需要，可以在这里进行额外的分析或预测
        # 例如：对单个样本进行预测
        if X_test is not None and len(X_test) > 0:
            sample_idx = 0
            sample = X_test.iloc[[sample_idx]]
            true_label = y_test.iloc[sample_idx] if y_test is not None else None
            
            prediction = model.predict(sample)[0]
            probability = model.predict_proba(sample)[0]
            
            print(f"\n示例预测:")
            print(f"  样本特征: {sample.shape}")
            print(f"  真实标签: {true_label}")
            print(f"  预测标签: {prediction}")
            print(f"  预测概率: [正常: {probability[0]:.3f}, 高危: {probability[1]:.3f}]")
    else:
        print("\n模型训练失败，请检查错误信息")
