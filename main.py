# main.py
import sys
import os
import time

# 添加 src 到路径，确保能导入模块
sys.path.insert(0, os.path.abspath("src"))

try:
    import src.config as config
    import src.data_loader as data_loader
    import src.feature_cluster as feature_cluster
    import src.model_catboost as model_class
    import src.model_regression as model_reg
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    print("请确保你在项目根目录下运行: python main.py")
    sys.exit(1)

def print_separator(title):
    print("\n" + "="*60)
    print(f"🚀 {title}")
    print("="*60)

def main():
    start_time = time.time()
    print_separator("Deep-SCL-Cat 系统启动 (Refactored Version)")
    
    # Step 1: 数据加载与清洗
    print_separator("Step 1: 数据加载与清洗")
    df_clean = data_loader.load_and_clean_data()
    if df_clean is None:
        print("❌ 数据加载失败，程序终止")
        return

    # Step 2: 聚类特征工程
    print_separator("Step 2: K-Means 聚类特征提取")
    df_clustered = feature_cluster.process_clustering()
    if df_clustered is None:
        print("❌ 聚类分析失败，程序终止")
        return

    # Step 3: 分类模型训练 (高危预警)
    print_separator("Step 3: 训练分类模型 (Is_High_Risk)")
    clf_model = model_class.train_classification_model()
    if clf_model is None:
        print("❌ 分类模型训练失败")

    # Step 4: 回归模型训练 (挂科数目预测)
    print_separator("Step 4: 训练回归模型 (Predict Failed Subjects)")
    reg_model = model_reg.train_regression_model()
    if reg_model is None:
        print("❌ 回归模型训练失败")

    # 总结
    end_time = time.time()
    duration = end_time - start_time
    
    print_separator("🎉 所有任务执行完毕")
    print(f"总耗时: {duration:.2f} 秒")
    print(f"输出目录: {config.OUTPUT_DIR}")
    print("\n现在你可以运行 'streamlit run app.py' 启动可视化界面了！")

if __name__ == "__main__":
    main()