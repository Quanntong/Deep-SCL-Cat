import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

print("=== 测试标准化器特征顺序修复 ===\n")

# 1. 加载标准化器
try:
    scaler = joblib.load('outputs/scaler.pkl')
    print("✅ 标准化器加载成功")
    
    # 检查标准化器是否有 feature_names_in_ 属性
    if hasattr(scaler, 'feature_names_in_'):
        scaler_features = list(scaler.feature_names_in_)
        print(f"✅ 标准化器特征名称: {scaler_features}")
        print(f"✅ 特征数量: {len(scaler_features)}")
    else:
        print("⚠️ 标准化器没有 feature_names_in_ 属性")
        # 尝试获取特征名称的其他方式
        if hasattr(scaler, 'get_feature_names_out'):
            scaler_features = list(scaler.get_feature_names_out())
            print(f"✅ 通过 get_feature_names_out 获取特征: {scaler_features}")
        else:
            print("❌ 无法获取标准化器特征名称")
            scaler_features = None
        
except Exception as e:
    print(f"❌ 标准化器加载失败: {e}")
    sys.exit(1)

# 2. 模拟输入数据
print("\n=== 模拟输入数据 ===")

# 从 config 获取 SCL-90 特征
try:
    import src.config as config
    scoring_features = config.SCL90_FEATS
    print(f"SCL-90 特征: {scoring_features}")
except:
    # 如果无法导入 config，使用硬编码的特征
    scoring_features = ['躯体化', '强迫', '人际', '抑郁', '焦虑', '敌对', '恐怖', '偏执', '精神', '其他']
    print(f"使用默认 SCL-90 特征: {scoring_features}")

# 创建测试数据
test_data = {}
for feature in scoring_features:
    test_data[feature] = [2.0]  # 默认值

df = pd.DataFrame(test_data)
print(f"输入数据列: {list(df.columns)}")
print(f"输入数据形状: {df.shape}")

# 3. 测试特征对齐
print("\n=== 测试特征对齐 ===")

if scaler_features is not None:
    # 检查列是否匹配
    missing_in_scaler = [col for col in scaler_features if col not in df.columns]
    missing_in_df = [col for col in df.columns if col not in scaler_features]
    
    print(f"标准化器中有但输入数据中缺失的列: {missing_in_scaler}")
    print(f"输入数据中有但标准化器中缺失的列: {missing_in_df}")
    
    # 如果标准化器特征顺序与输入数据不同，重新排列
    if list(df.columns) != scaler_features:
        print("⚠️ 特征顺序不匹配，重新排列...")
        # 确保所有标准化器特征都在输入数据中
        for col in scaler_features:
            if col not in df.columns:
                print(f"  补零缺失列: {col}")
                df[col] = 0
        
        # 重新排列以匹配标准化器顺序
        df_aligned = df[scaler_features].copy()
        print(f"✅ 对齐后的列: {list(df_aligned.columns)}")
        df = df_aligned
    else:
        print("✅ 特征顺序已匹配")
else:
    print("⚠️ 无法获取标准化器特征，跳过对齐")

# 4. 测试标准化
print("\n=== 测试标准化 ===")
try:
    X_scaled = scaler.transform(df)
    print(f"✅ 标准化成功！")
    print(f"标准化后形状: {X_scaled.shape}")
    print(f"标准化后数据示例 (前5个特征): {X_scaled[0, :5]}")
    
except Exception as e:
    print(f"❌ 标准化失败: {e}")
    error_msg = str(e)
    
    if "feature names" in error_msg.lower() or "feature order" in error_msg.lower():
        print("⚠️ 检测到特征名称/顺序错误！")
        print("可能的原因:")
        print(f"1. 标准化器期望的特征: {scaler_features if scaler_features else '未知'}")
        print(f"2. 实际提供的特征: {list(df.columns)}")
    elif "shape" in error_msg.lower():
        print("⚠️ 数据形状错误！")
    else:
        print("⚠️ 其他类型的错误")

print("\n=== 测试完成 ===")
