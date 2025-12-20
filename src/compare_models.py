import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer  # <--- 新增：引入填充工具
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
import warnings
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

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 忽略警告
warnings.filterwarnings('ignore')


def run_comparison(X_train, y_train, X_test, y_test, our_model):
    """
    运行多模型对比分析
    """
    print("=" * 60)
    print("多模型性能对比分析")
    print("=" * 60)
    
    # 确保输出目录存在
    outputs_dir = os.path.join(config.BASE_DIR, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    # 定义基线模型
    baselines = {}
    
    # 1. 随机森林 (Random Forest) - Sklearn 模型不支持 NaN，需要 Imputer
    try:
        baselines["Random Forest"] = make_pipeline(
            SimpleImputer(strategy='median'),  # <--- 新增：自动填补空值
            RandomForestClassifier(
                n_estimators=100, 
                random_state=config.RANDOM_SEED,
                n_jobs=-1
            )
        )
        print("✓ 已添加 Random Forest 模型")
    except Exception as e:
        print(f"✗ Random Forest 模型添加失败: {e}")
    
    # 2. XGBoost (原生支持 NaN，通常不需要 Imputer，但加上也无妨)
    try:
        from xgboost import XGBClassifier
        baselines["XGBoost"] = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=config.RANDOM_SEED,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
        print("✓ 已添加 XGBoost 模型")
    except ImportError:
        print("✗ XGBoost 未安装，跳过此模型")
    except Exception as e:
        print(f"✗ XGBoost 模型添加失败: {e}")
    
    # 3. LightGBM (原生支持 NaN)
    try:
        from lightgbm import LGBMClassifier
        baselines["LightGBM"] = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=config.RANDOM_SEED,
            n_jobs=-1
        )
        print("✓ 已添加 LightGBM 模型")
    except ImportError:
        print("✗ LightGBM 未安装，跳过此模型")
    except Exception as e:
        print(f"✗ LightGBM 模型添加失败: {e}")
    
    # 4. 逻辑回归 (Logistic Regression) - 必须处理 NaN
    try:
        baselines["Logistic Regression"] = make_pipeline(
            SimpleImputer(strategy='median'),  # <--- 新增：自动填补空值
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                random_state=config.RANDOM_SEED,
                n_jobs=-1
            )
        )
        print("✓ 已添加 Logistic Regression 模型")
    except Exception as e:
        print(f"✗ Logistic Regression 模型添加失败: {e}")
    
    # 5. SVM - 必须处理 NaN
    try:
        baselines["SVM"] = make_pipeline(
            SimpleImputer(strategy='median'),  # <--- 新增：自动填补空值
            StandardScaler(),
            SVC(
                probability=True,
                random_state=config.RANDOM_SEED
            )
        )
        print("✓ 已添加 SVM 模型")
    except Exception as e:
        print(f"✗ SVM 模型添加失败: {e}")
    
    print(f"\n总共添加了 {len(baselines)} 个基线模型")
    
    # 存储结果
    results = []
    
    # 训练和评估基线模型
    print("\n" + "-" * 60)
    print("开始训练和评估基线模型...")
    print("-" * 60)
    
    for model_name, model in baselines.items():
        try:
            print(f"\n训练 {model_name}...")
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            y_pred_proba = None
            
            # 尝试获取预测概率（用于AUC计算）
            try:
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_pred_proba = model.decision_function(X_test)
            except:
                y_pred_proba = None
            
            # 计算指标
            recall = recall_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            accuracy = accuracy_score(y_test, y_pred)
            
            # 计算AUC
            auc = np.nan
            if y_pred_proba is not None:
                try:
                    auc = roc_auc_score(y_test, y_pred_proba)
                except:
                    auc = np.nan
            
            # 存储结果
            results.append({
                'Model': model_name,
                'Recall': recall,
                'Precision': precision,
                'F1': f1,
                'Accuracy': accuracy,
                'AUC': auc
            })
            
            print(f"  Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
            
        except Exception as e:
            print(f"  ✗ {model_name} 训练或评估失败: {e}")
            results.append({
                'Model': model_name,
                'Recall': np.nan, 'Precision': np.nan, 'F1': np.nan, 'Accuracy': np.nan, 'AUC': np.nan
            })
    
    # 评估我们的模型（Deep-SCL-Cat）
    print("\n" + "-" * 60)
    print("评估 Deep-SCL-Cat 模型...")
    print("-" * 60)
    
    try:
        # 使用我们的模型进行预测
        y_pred_our = our_model.predict(X_test)
        y_pred_proba_our = our_model.predict_proba(X_test)[:, 1]
        
        # 计算指标
        recall_our = recall_score(y_test, y_pred_our, average='weighted')
        precision_our = precision_score(y_test, y_pred_our, average='weighted')
        f1_our = f1_score(y_test, y_pred_our, average='weighted')
        accuracy_our = accuracy_score(y_test, y_pred_our)
        auc_our = roc_auc_score(y_test, y_pred_proba_our)
        
        results.append({
            'Model': 'Deep-SCL-Cat (Ours)',
            'Recall': recall_our,
            'Precision': precision_our,
            'F1': f1_our,
            'Accuracy': accuracy_our,
            'AUC': auc_our
        })
        
        print(f"  Recall: {recall_our:.4f}, F1: {f1_our:.4f}, Accuracy: {accuracy_our:.4f}, AUC: {auc_our:.4f}")
        
    except Exception as e:
        print(f"  ✗ Deep-SCL-Cat 模型评估失败: {e}")
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(results)
    
    # 按Recall降序排序
    results_df_sorted = results_df.sort_values('Recall', ascending=False).reset_index(drop=True)
    
    # 保存结果到CSV
    csv_path = os.path.join(outputs_dir, 'model_comparison.csv')
    results_df_sorted.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ 对比结果已保存到: {csv_path}")
    
    # 可视化
    print("\n" + "-" * 60)
    print("生成可视化图表...")
    print("-" * 60)
    
    try:
        plot_data = results_df_sorted.copy()
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('多模型性能对比分析', fontsize=16, fontweight='bold')
        
        metrics = ['Recall', 'F1', 'Accuracy', 'AUC']
        metric_titles = ['召回率 (Recall)', 'F1分数', '准确率 (Accuracy)', 'AUC']
        
        for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
            ax = axes[idx // 2, idx % 2]
            plot_metric = plot_data.dropna(subset=[metric]).sort_values(metric, ascending=True)
            
            bars = ax.barh(plot_metric['Model'], plot_metric[metric], 
                          color=plt.cm.Set3(np.linspace(0, 1, len(plot_metric))))
            
            for i, (model_name, bar) in enumerate(zip(plot_metric['Model'], bars)):
                if 'Deep-SCL-Cat' in model_name:
                    bar.set_color('red')
                    bar.set_edgecolor('darkred')
                    bar.set_linewidth(2)
            
            for i, (value, bar) in enumerate(zip(plot_metric[metric], bars)):
                ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{value:.3f}', va='center', ha='left', fontsize=9)
            
            ax.set_xlabel(title, fontsize=12)
            ax.set_xlim(0, min(1.0, plot_metric[metric].max() * 1.2))
            ax.grid(True, alpha=0.3)
            ax.xaxis.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        img_path = os.path.join(outputs_dir, 'model_comparison.png')
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        print(f"✓ 可视化图表已保存到: {img_path}")
        
    except Exception as e:
        print(f"✗ 可视化生成失败: {e}")
    
    print("\n" + "=" * 60)
    print("模型对比总结")
    print("=" * 60)
    print(results_df_sorted.to_string(index=False))
    
    return results_df_sorted

if __name__ == "__main__":
    print("Comparison module loaded.")