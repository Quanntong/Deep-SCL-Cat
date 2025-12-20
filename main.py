#!/usr/bin/env python3
"""
Deep-SCL-Cat 项目主入口文件
串联整个工作流：数据加载 → 聚类特征工程 → 模型训练 → 策略寻优 → 解释性分析
"""

import sys
import os
import time

# 添加src目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_step_header(step_num, step_name):
    """打印步骤头部信息"""
    print("\n" + "=" * 60)
    print(f">>> Step {step_num}: {step_name}")
    print("=" * 60)

def print_step_footer(step_num, step_name, elapsed_time):
    """打印步骤尾部信息"""
    print(f"\n✓ Step {step_num}: {step_name} 完成 (耗时: {elapsed_time:.2f}秒)")
    print("-" * 60)

def main():
    """
    主函数：执行完整的Deep-SCL-Cat工作流
    """
    print("=" * 60)
    print("Deep-SCL-Cat 工作流启动")
    print("=" * 60)
    print("项目描述: SCL-90心理评估数据的CatBoost分类与可解释性分析")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    total_start_time = time.time()
    
    # Step 1: 数据加载
    step_start_time = time.time()
    print_step_header(1, "数据加载")
    
    try:
        import src.data_loader as data_loader
        print("导入 data_loader 模块成功")
        
        # 调用数据加载函数（仅打印信息，无需保存返回值）
        print("\n执行数据加载与清洗...")
        cleaned_data = data_loader.load_and_clean_data()
        
        if cleaned_data is not None:
            print(f"数据加载成功，形状: {cleaned_data.shape}")
            print(f"数据列名: {list(cleaned_data.columns)}")
        else:
            print("警告: 数据加载返回None，但流程将继续")
            
    except ImportError as e:
        print(f"错误: 无法导入 data_loader 模块 - {e}")
        return
    except Exception as e:
        print(f"数据加载过程中发生错误: {e}")
        print("警告: 数据加载失败，但流程将继续")
    
    step_elapsed = time.time() - step_start_time
    print_step_footer(1, "数据加载", step_elapsed)
    
    # Step 2: 聚类特征工程
    step_start_time = time.time()
    print_step_header(2, "聚类特征工程")
    
    try:
        import src.feature_cluster as feature_cluster
        print("导入 feature_cluster 模块成功")
        
        # 调用聚类特征工程函数
        print("\n执行K-Means聚类与特征增强...")
        result = feature_cluster.process_clustering()
        
        if result is not None:
            df_clustered, kmeans, X_scaled, feature_columns = result
            print(f"聚类完成，数据形状: {df_clustered.shape}")
            print(f"聚类标签分布: {df_clustered['Cluster_Label'].value_counts().to_dict()}")
            
            # 分析聚类结果
            feature_cluster.analyze_clusters(df_clustered, kmeans, X_scaled, feature_columns)
        else:
            print("警告: 聚类特征工程返回None，但流程将继续")
            
    except ImportError as e:
        print(f"错误: 无法导入 feature_cluster 模块 - {e}")
        return
    except Exception as e:
        print(f"聚类特征工程过程中发生错误: {e}")
        print("警告: 聚类特征工程失败，但流程将继续")
    
    step_elapsed = time.time() - step_start_time
    print_step_footer(2, "聚类特征工程", step_elapsed)
    
    # Step 3: 模型训练
    step_start_time = time.time()
    print_step_header(3, "模型训练")
    
    try:
        import src.model_catboost as model_catboost
        print("导入 model_catboost 模块成功")
        
        # 调用模型训练函数
        print("\n执行CatBoost模型训练...")
        model, X_train, X_test, y_train, y_test = model_catboost.train_model()
        
        if model is not None:
            print(f"模型训练成功，树数量: {model.tree_count_}")
            print(f"训练集形状: {X_train.shape if X_train is not None else 'N/A'}")
            print(f"测试集形状: {X_test.shape if X_test is not None else 'N/A'}")
        else:
            print("警告: 模型训练返回None，但流程将继续")
            
    except ImportError as e:
        print(f"错误: 无法导入 model_catboost 模块 - {e}")
        return
    except Exception as e:
        print(f"模型训练过程中发生错误: {e}")
        print("警告: 模型训练失败，但流程将继续")
    
    step_elapsed = time.time() - step_start_time
    print_step_footer(3, "模型训练", step_elapsed)
    
    # Step 4: 策略寻优
    step_start_time = time.time()
    print_step_header(4, "策略寻优")
    
    try:
        import src.strategy_recall as strategy_recall
        print("导入 strategy_recall 模块成功")
        
        # 检查模型文件是否存在
        model_path = os.path.join('outputs', 'catboost_model.cbm')
        if os.path.exists(model_path):
            print(f"模型文件存在: {model_path}")
            
            # 调用策略寻优函数
            print("\n执行阈值寻优策略...")
            optimal_threshold = strategy_recall.find_optimal_threshold()
            
            if optimal_threshold is not None:
                print(f"最佳阈值: {optimal_threshold:.4f}")
                print(f"使用此阈值进行二分类预测:")
                print(f"  如果样本属于'高危'的概率 > {optimal_threshold:.4f}，则预测为'高危(1)'")
                print(f"  否则预测为'正常(0)'")
            else:
                print("警告: 策略寻优返回None，但流程将继续")
        else:
            print(f"警告: 模型文件不存在 - {model_path}")
            print("跳过策略寻优步骤")
            
    except ImportError as e:
        print(f"错误: 无法导入 strategy_recall 模块 - {e}")
        return
    except Exception as e:
        print(f"策略寻优过程中发生错误: {e}")
        print("警告: 策略寻优失败，但流程将继续")
    
    step_elapsed = time.time() - step_start_time
    print_step_footer(4, "策略寻优", step_elapsed)
    
    # Step 5: 解释性分析
    step_start_time = time.time()
    print_step_header(5, "解释性分析")
    
    try:
        import src.explainability as explainability
        print("导入 explainability 模块成功")
        
        # 检查模型文件是否存在
        model_path = os.path.join('outputs', 'catboost_model.cbm')
        if os.path.exists(model_path):
            print(f"模型文件存在: {model_path}")
            
            # 调用解释性分析函数
            print("\n执行SHAP可解释性分析...")
            explainability.explain_model()
        else:
            print(f"警告: 模型文件不存在 - {model_path}")
            print("跳过解释性分析步骤")
            
    except ImportError as e:
        print(f"错误: 无法导入 explainability 模块 - {e}")
        return
    except Exception as e:
        print(f"解释性分析过程中发生错误: {e}")
        print("警告: 解释性分析失败，但流程将继续")
    
    step_elapsed = time.time() - step_start_time
    print_step_footer(5, "解释性分析", step_elapsed)
    
    # Step 6: 多模型对比
    step_start_time = time.time()
    print_step_header(6, "多模型对比")
    
    try:
        import src.compare_models as compare_models
        print("导入 compare_models 模块成功")
        
        # 检查是否有训练好的模型和数据
        if model is not None and X_train is not None and X_test is not None:
            print("\n执行多模型性能对比分析...")
            print("对比模型: Random Forest, XGBoost, LightGBM, Logistic Regression, SVM")
            print("核心指标: Recall (召回率)")
            
            # 调用对比函数
            comparison_results = compare_models.run_comparison(
                X_train, y_train, X_test, y_test, model
            )
            
            if comparison_results is not None:
                print(f"对比分析完成，共评估 {len(comparison_results)} 个模型")
                print(f"结果已保存到 outputs/model_comparison.csv 和 outputs/model_comparison.png")
            else:
                print("警告: 对比分析返回None，但流程将继续")
        else:
            print("警告: 缺少模型或数据，跳过对比分析步骤")
            print(f"  模型: {'可用' if model is not None else '不可用'}")
            print(f"  训练数据: {'可用' if X_train is not None else '不可用'}")
            print(f"  测试数据: {'可用' if X_test is not None else '不可用'}")
            
    except ImportError as e:
        print(f"错误: 无法导入 compare_models 模块 - {e}")
        print("跳过对比分析步骤")
    except Exception as e:
        print(f"对比分析过程中发生错误: {e}")
        print("警告: 对比分析失败，但流程将继续")
    
    step_elapsed = time.time() - step_start_time
    print_step_footer(6, "多模型对比", step_elapsed)
    
    # 工作流完成
    total_elapsed = time.time() - total_start_time
    
    print("\n" + "=" * 60)
    print("Deep-SCL-Cat 工作流完成!")
    print("=" * 60)
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {total_elapsed:.2f}秒 ({total_elapsed/60:.2f}分钟)")
    print("\n生成的文件:")
    print("-" * 40)
    
    # 列出生成的文件
    outputs_dir = 'outputs'
    if os.path.exists(outputs_dir):
        files = os.listdir(outputs_dir)
        for i, file in enumerate(sorted(files), 1):
            file_path = os.path.join(outputs_dir, file)
            file_size = os.path.getsize(file_path)
            print(f"{i:2d}. {file:30s} ({file_size:,} bytes)")
    else:
        print("outputs目录不存在")
    
    

if __name__ == "__main__":
    # 主程序入口
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n工作流被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n工作流执行过程中发生未预期错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
