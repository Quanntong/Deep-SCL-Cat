import pandas as pd
import numpy as np
import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import src.config as config
except ImportError:
    try:
        from . import config
    except ImportError:
        import config

def load_and_clean_data(filepath=None):
    print("\n" + "="*60)
    print("🛡️ 正在运行【增强白名单模式】Data Loader！智能处理数据列... 🛡️")
    print("="*60 + "\n")

    if filepath is None:
        filepath = os.path.join(config.DATA_RAW, config.RAW_FILE)
    
    try:
        # encoding='utf-8-sig' 专门解决 \ufeff 这种看不见的幽灵字符
        df = pd.read_csv(filepath, encoding='utf-8-sig') 
        print(f"✅ 原始数据加载: {df.shape}")
    except:
        # 如果 utf-8-sig 失败，尝试默认读取
        try:
            df = pd.read_csv(filepath)
            print("⚠️ 使用默认编码读取...")
        except Exception as e:
            print(f"❌ 读取失败: {e}")
            return None

    # 1. 增强列名清洗：去除所有不可见字符和多余空格
    df.columns = df.columns.astype(str).str.strip()
    # 去除所有空白字符（包括换行、制表符等）
    df.columns = df.columns.str.replace(r'[\n\r\t\s]+', '', regex=True)
    # 去除引号和其他特殊字符
    df.columns = df.columns.str.replace(r'[\"\'`]', '', regex=True)
    # 去除点号后面的数字（如"精神.1" -> "精神"）
    df.columns = df.columns.str.replace(r'\.\d+$', '', regex=True)
    
    print(f"🔧 清洗后列名: {list(df.columns)}")

    # ==========================================
    # 💎 核心大招：增强白名单机制 (Enhanced White-listing)
    # ==========================================
    print("\n💎 [Step 1] 启动智能VIP白名单过滤...")
    
    # 我们只允许这些列存在，其他的统统不要！
    # 包含：config里定义的10个心理因子 + 年龄/性别 + 标签
    vip_list = config.SCL90_FEATS + ['年龄', '性别', 'Risk_Label', 'Cluster_Label']
    
    # 找出数据里实际存在的VIP列
    valid_cols = [c for c in df.columns if c in vip_list]
    
    # 检查缺失的SCL-90特征
    missing_scl_feats = [feat for feat in config.SCL90_FEATS if feat not in valid_cols]
    
    # 处理缺失的特征（特别是"其他"列）
    for missing_feat in missing_scl_feats:
        print(f"   ⚠️  检测到缺失特征: '{missing_feat}'")
        if missing_feat == '其他':
            # 对于"其他"列，用0填充或使用默认值
            print(f"   🔧  自动创建 '{missing_feat}' 列，使用默认值2.0")
            df[missing_feat] = 2.0  # SCL-90的中性值
            valid_cols.append(missing_feat)
        else:
            print(f"   ⚠️  特征 '{missing_feat}' 在数据中不存在，可能影响模型性能")
    
    # 处理重复列问题：合并相同名称的列（如"精神"、"精神.1"等清洗后都变成"精神"）
    # 首先找出所有重复的列名
    from collections import defaultdict
    col_groups = defaultdict(list)
    for col in df.columns:
        col_groups[col].append(col)
    
    # 对于有重复的列，合并数据
    for col_name, original_cols in col_groups.items():
        if len(original_cols) > 1:
            print(f"   🔧  检测到重复列 '{col_name}'，原始列: {original_cols}")
            # 合并策略：优先使用第一个非空值
            if col_name in df.columns:
                # 如果已经存在该列（清洗后），需要合并数据
                # 首先，我们需要确保有多个不同的列（不仅仅是同一个列的多个引用）
                # 由于列名清洗后都变成了'精神'，我们需要跟踪原始列
                # 创建一个新的Series来合并所有数据
                merged_series = pd.Series(dtype=float)
                
                # 收集所有列的数据
                all_data = []
                for orig_col in original_cols:
                    if orig_col in df.columns:
                        all_data.append(df[orig_col])
                
                if all_data:
                    # 合并数据：使用第一个非NaN值
                    merged_series = all_data[0].copy()
                    for i in range(1, len(all_data)):
                        # 用后续列的数据填充缺失值
                        mask = merged_series.isna() & all_data[i].notna()
                        try:
                            mask_count = int(mask.sum())
                        except:
                            mask_count = 0
                        if mask_count > 0:
                            merged_series[mask] = all_data[i][mask]
                            print(f"   🔧  用第{i+1}个'{col_name}'列填充了 {mask_count} 个缺失值")
                
                # 更新DataFrame
                df[col_name] = merged_series
    
    # 只保留有效的列（去除重复列，只保留合并后的列）
    df = df[[c for c in valid_cols if c in df.columns or c == '其他']]
    
    # 去除重复列（如果有多个相同列名的列）
    df = df.loc[:, ~df.columns.duplicated()]
    
    # 确保所有SCL90特征都存在
    for feat in config.SCL90_FEATS:
        if feat not in df.columns:
            if feat == '其他':
                df[feat] = 2.0  # 默认值
            else:
                # 对于其他缺失特征，用中位数填充
                print(f"   ⚠️  特征 '{feat}' 仍然缺失，用0填充")
                df[feat] = 0.0
    
    print(f"   ✅  最终保留 {len(df.columns)} 个有效列: {list(df.columns)}")
    
    # ==========================================
    # 💉 2. 缺失值填充与类型转换
    # ==========================================
    print("\n💉 [Step 2] 缺失值填充与类型转换...")
    for col in df.columns:
        if col not in ['Risk_Label', 'Cluster_Label']:
            # 转换为数值类型
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # 填充缺失值
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_value = df[col].median() if not df[col].isnull().all() else 0
                df[col] = df[col].fillna(fill_value)
                print(f"   🔧  '{col}' 列: 填充 {df[col].isnull().sum()} 个缺失值为 {fill_value:.2f}")
            else:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 0
                df[col] = df[col].fillna(mode_val)
    
    # 验证所有SCL90特征都存在且为数值类型
    print("\n📊 [Step 3] 数据验证...")
    scl_feats_present = [feat for feat in config.SCL90_FEATS if feat in df.columns]
    print(f"   ✅  SCL-90特征存在 {len(scl_feats_present)}/{len(config.SCL90_FEATS)} 个")
    
    if len(scl_feats_present) < len(config.SCL90_FEATS):
        missing = [feat for feat in config.SCL90_FEATS if feat not in df.columns]
        print(f"   ⚠️  缺失特征: {missing}")
    
    print(f"\n✅ 最终纯净数据形状: {df.shape}")
    print(f"✅ 数据列: {list(df.columns)}")
    return df

if __name__ == "__main__":
    load_and_clean_data()
