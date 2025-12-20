import pandas as pd
import shap
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# 配置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import src.config as config
except ImportError:
    try:
        from . import config
    except ImportError:
        import config

def explain_model():
    print("=" * 50)
    print("SHAP模型可解释性分析")
    print("=" * 50)
    plt.switch_backend('Agg')
    
    # 1. 加载模型
    model_path = os.path.join(config.BASE_DIR, 'outputs', 'catboost_model.cbm')
    try:
        model = CatBoostClassifier()
        model.load_model(model_path)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 2. 加载数据
    processed_file_path = os.path.join(config.DATA_PROCESSED, config.PROCESSED_FILE)
    try:
        df = pd.read_csv(processed_file_path)
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    # 3. 数据准备（关键步骤：剔除无关列！）
    drop_cols = ['Risk_Label', '姓名', '学号', 'id', 'avg_score']
    
    # 确保 Cluster_Label 是数字
    if 'Cluster_Label' in df.columns:
        df['Cluster_Label'] = df['Cluster_Label'].fillna(0).astype(int)

    # 生成纯净的特征矩阵 X
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    print(f"原始特征列 ({len(X.columns)}): {list(X.columns)}")
    
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
    
    print(f"最终用于解释的特征列 ({len(X.columns)}): {list(X.columns)}")
    
    # 4. SHAP 计算
    try:
        explainer = shap.TreeExplainer(model)
        # 采样部分数据加速
        X_sample = X.sample(n=min(500, len(X)), random_state=42)
        shap_values = explainer.shap_values(X_sample)
    except Exception as e:
        print(f"SHAP计算失败: {e}")
        return

    # 5. 保存图表
    outputs_dir = os.path.join(config.BASE_DIR, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    # 摘要图
    try:
        plt.figure()
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.savefig(os.path.join(outputs_dir, 'shap_summary_dot.png'), bbox_inches='tight')
        plt.close()
        print("SHAP 摘要图已保存")
    except Exception as e:
        print(f"绘图失败: {e}")

    # 特征重要性 CSV 和条形图
    try:
        if isinstance(shap_values, list): # 多分类
            vals = np.abs(shap_values[0]).mean(0)
        else: # 二分类
            vals = np.abs(shap_values).mean(0)
            
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            '平均绝对SHAP值': vals
        }).sort_values('平均绝对SHAP值', ascending=False)
        
        importance_df.to_csv(os.path.join(outputs_dir, 'shap_feature_importance.csv'), index=False)
        print("特征重要性 CSV 已保存")
        print("\nTop 5 重要特征:")
        print(importance_df.head(5))
        
        # 生成特征重要性条形图
        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
        bars = plt.barh(range(len(importance_df)), importance_df['平均绝对SHAP值'], color=colors)
        plt.yticks(range(len(importance_df)), importance_df['Feature'], fontsize=10)
        plt.xlabel('平均绝对SHAP值', fontsize=12)
        plt.title('SHAP 特征重要性排序', fontsize=14, pad=20)
        plt.gca().invert_yaxis()  # 最重要的在顶部
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, importance_df['平均绝对SHAP值'])):
            plt.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{val:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(outputs_dir, 'shap_importance_bar.png'), bbox_inches='tight', dpi=300)
        plt.close()
        print("SHAP 特征重要性条形图已保存")
        
    except Exception as e:
        print(f"保存重要性数据失败: {e}")

if __name__ == "__main__":
    explain_model()
