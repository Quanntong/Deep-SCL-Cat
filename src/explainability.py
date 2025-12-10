import pandas as pd
import shap
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# 配置中文字体（针对 Windows 环境）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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


def explain_model():
    """
    使用SHAP进行模型可解释性分析
    
    功能：
    1. 加载训练好的CatBoost模型和预处理数据
    2. 计算SHAP值
    3. 生成特征重要性可视化
    4. 分析Top 5最重要的风险因子
    """
    print("=" * 50)
    print("SHAP模型可解释性分析")
    print("=" * 50)
    
    # 配置Matplotlib使用非交互后端
    plt.switch_backend('Agg')
    print("Matplotlib已配置为非交互后端")
    
    # 1. 加载资源
    print("\n步骤1: 加载模型和数据...")
    
    # 加载模型
    model_path = os.path.join(config.BASE_DIR, 'outputs', 'catboost_model.cbm')
    try:
        model = CatBoostClassifier()
        model.load_model(model_path)
        print(f"模型加载成功: {model_path}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 加载预处理数据
    processed_file_path = os.path.join(config.DATA_PROCESSED, config.PROCESSED_FILE)
    try:
        df = pd.read_csv(processed_file_path)
        print(f"数据加载成功，形状: {df.shape}")
        print(f"数据列名: {list(df.columns)}")
    except FileNotFoundError:
        print(f"错误: 文件未找到 - {processed_file_path}")
        print("请先运行 feature_cluster.py 生成聚类数据")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return
    
    # 2. 数据准备
    print("\n步骤2: 数据准备...")
    
    # 检查是否存在目标列 'Risk_Label'
    if 'Risk_Label' not in df.columns:
        print("数据中不存在 'Risk_Label' 列，将随机生成二分类目标变量用于演示")
        print("注意: 在实际应用中，请确保数据包含真实的目标变量")
        
        # 随机生成二分类目标变量，模拟高危/正常标签
        # 这里使用简单的规则：基于 score 列的和来生成标签
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
    
    # 定义特征和目标
    target_column = 'Risk_Label'
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    print(f"特征数据形状: {X.shape}")
    print(f"目标变量形状: {y.shape}")
    
    # 3. SHAP计算
    print("\n步骤3: 计算SHAP值...")
    
    try:
        # 初始化SHAP解释器
        explainer = shap.TreeExplainer(model)
        print("SHAP解释器初始化成功")
        
        # 计算SHAP值
        # 注意：对于大型数据集，可以计算部分样本的SHAP值以提高性能
        if len(X) > 1000:
            print(f"数据量较大 ({len(X)} 个样本)，将随机采样1000个样本计算SHAP值")
            sample_indices = X.sample(n=min(1000, len(X)), random_state=config.RANDOM_SEED).index
            X_sample = X.loc[sample_indices]
            shap_values = explainer.shap_values(X_sample)
            print(f"已计算 {len(X_sample)} 个样本的SHAP值")
        else:
            shap_values = explainer.shap_values(X)
            print(f"已计算 {len(X)} 个样本的SHAP值")
        
        # 获取期望值
        expected_value = explainer.expected_value
        print(f"SHAP期望值: {expected_value}")
        
    except Exception as e:
        print(f"SHAP计算失败: {e}")
        return
    
    # 4. 可视化输出
    print("\n步骤4: 生成可视化图表...")
    
    # 确保输出目录存在
    outputs_dir = os.path.join(config.BASE_DIR, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    # 绘制 Summary Plot (条形图模式) - 特征重要性排名
    try:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample if 'X_sample' in locals() else X, 
                         plot_type="bar", show=False)
        bar_chart_path = os.path.join(outputs_dir, 'shap_importance_bar.png')
        plt.savefig(bar_chart_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"特征重要性条形图已保存: {bar_chart_path}")
    except Exception as e:
        print(f"生成特征重要性条形图失败: {e}")
    
    # 绘制 Summary Plot (散点图模式) - 特征值高低对风险的影响方向
    try:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample if 'X_sample' in locals() else X, 
                         show=False)
        dot_chart_path = os.path.join(outputs_dir, 'shap_summary_dot.png')
        plt.savefig(dot_chart_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"SHAP摘要散点图已保存: {dot_chart_path}")
    except Exception as e:
        print(f"生成SHAP摘要散点图失败: {e}")
    
    # 5. 关键特征分析
    print("\n步骤5: 关键特征分析...")
    
    try:
        # 计算每个特征的平均绝对SHAP值
        if isinstance(shap_values, list):
            # 对于多分类问题，SHAP值是一个列表
            shap_abs_mean = np.abs(shap_values[1]).mean(axis=0)  # 使用正类（高危）的SHAP值
        else:
            # 对于二分类问题，SHAP值是一个数组
            shap_abs_mean = np.abs(shap_values).mean(axis=0)
        
        # 创建特征重要性DataFrame
        feature_importance_df = pd.DataFrame({
            '特征': X.columns,
            '平均绝对SHAP值': shap_abs_mean
        })
        
        # 按重要性排序
        feature_importance_df = feature_importance_df.sort_values('平均绝对SHAP值', ascending=False)
        
        # 打印Top 5最重要的风险因子
        print("\nTop 5 最重要的风险因子:")
        print("=" * 60)
        for i, (_, row) in enumerate(feature_importance_df.head(5).iterrows(), 1):
            print(f"{i}. {row['特征']}: {row['平均绝对SHAP值']:.6f}")
        print("=" * 60)
        
        # 保存特征重要性到CSV文件
        importance_csv_path = os.path.join(outputs_dir, 'shap_feature_importance.csv')
        feature_importance_df.to_csv(importance_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n特征重要性已保存到: {importance_csv_path}")
        
        # 打印所有特征的重要性
        print(f"\n所有特征重要性 (共 {len(feature_importance_df)} 个特征):")
        print(feature_importance_df.to_string(index=False))
        
    except Exception as e:
        print(f"关键特征分析失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. 额外分析：单个特征的SHAP依赖图
    print("\n步骤6: 生成关键特征的SHAP依赖图...")
    
    try:
        # 检查feature_importance_df是否存在
        if 'feature_importance_df' in locals():
            # 选择Top 3特征生成依赖图
            top_features = feature_importance_df.head(3)['特征'].tolist()
            
            for i, feature in enumerate(top_features):
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(feature, shap_values, 
                                   X_sample if 'X_sample' in locals() else X,
                                   show=False)
                dep_plot_path = os.path.join(outputs_dir, f'shap_dependence_{feature}.png')
                plt.savefig(dep_plot_path, bbox_inches='tight', dpi=300)
                plt.close()
                print(f"  {feature} 的SHAP依赖图已保存: {dep_plot_path}")
        else:
            print("  无法生成依赖图：特征重要性分析失败")
    except Exception as e:
        print(f"生成SHAP依赖图失败: {e}")
    
    print("\n" + "=" * 50)
    print("SHAP可解释性分析完成!")
    print("=" * 50)
    print(f"生成的文件保存在: {outputs_dir}")
    print("  1. shap_importance_bar.png - 特征重要性条形图")
    print("  2. shap_summary_dot.png - SHAP摘要散点图")
    print("  3. shap_feature_importance.csv - 特征重要性CSV文件")
    if 'top_features' in locals() and top_features:
        for feature in top_features:
            print(f"  4. shap_dependence_{feature}.png - {feature}的SHAP依赖图")


if __name__ == "__main__":
    # 主程序入口
    print("=" * 50)
    print("SHAP模型可解释性分析模块")
    print("=" * 50)
    
    # 执行SHAP分析函数
    explain_model()
