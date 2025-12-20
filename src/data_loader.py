import os
import sys
import re
import pandas as pd
import numpy as np

# 确保能导入 config
try:
    import src.config as config
except ImportError:
    # 兼容直接运行此脚本的情况
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config

def _clean_column_names(df):
    """
    清洗列名：去除空白、特殊字符及Pandas自动添加的后缀（如 .1）
    """
    def clean_name(name):
        name = str(name).strip()
        # 去除不可见字符、引号
        name = re.sub(r'[\s"\'`\n\r\t]+', '', name)
        # 去除Pandas读取重复列时自动产生的后缀 (e.g., "精神.1" -> "精神")
        name = re.sub(r'\.\d+$', '', name)
        return name

    df.columns = [clean_name(c) for c in df.columns]
    return df

def _handle_duplicate_columns(df):
    """
    合并重复列：如果有多个同名列（如两个'精神'），合并它们的数据
    逻辑：优先取第一列的值，如果为空则用后续列填充
    """
    if not df.columns.has_duplicates:
        return df
    
    print(f"Merge Log: Detected duplicate columns: {df.columns[df.columns.duplicated()].unique()}")
    
    # 使用 groupby + bfill (向后填充) + iloc 取第一列
    # 这比手动写循环快得多且不易出错
    return df.groupby(df.columns, axis=1).apply(lambda x: x.bfill(axis=1).iloc[:, 0])

def _enforce_schema(df):
    """
    强制对齐数据模式（Schema Enforcement）
    1. 仅保留 VIP 列
    2. 补充缺失列
    """
    # 定义白名单列
    target_cols = config.SCL90_FEATS + ['年龄', '性别', 'Risk_Label', 'Cluster_Label']
    
    # 1. 过滤：只保留存在的 VIP 列 (先取交集，防止 KeyError)
    valid_cols = [c for c in target_cols if c in df.columns]
    df = df[valid_cols]
    
    # 2. 补全：检查缺失列并处理
    missing_cols = set(target_cols) - set(df.columns)
    
    if missing_cols:
        print(f"Schema Log: Missing columns found: {missing_cols}")
        for col in missing_cols:
            if col == '其他':
                # 特殊业务逻辑：'其他'列默认填充 2.0
                df[col] = 2.0
            else:
                # 其他缺失特征暂填 NaN，后续统一处理
                df[col] = np.nan
                
    # 确保列顺序一致
    return df.reindex(columns=target_cols)

def load_and_clean_data(filepath=None):
    """
    主数据加载与清洗流程
    """
    if filepath is None:
        filepath = os.path.join(config.DATA_RAW, config.RAW_FILE)

    print("-" * 40)
    print(f"Loading data from: {os.path.basename(filepath)}")

    # 1. 加载数据
    try:
        # 优先尝试 utf-8-sig (Excel常见编码)
        df = pd.read_csv(filepath, encoding='utf-8-sig')
    except UnicodeDecodeError:
        print("Encoding Warning: 'utf-8-sig' failed, trying default encoding.")
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error: Failed to read file. {e}")
        return None

    print(f"Raw shape: {df.shape}")

    # 2. 清洗列名
    df = _clean_column_names(df)

    # 3. 处理重复列 (核心逻辑)
    # 原数据中可能存在 "精神" 和 "精神.1"，清洗后会变成两个 "精神"
    # 这里需要将其合并，防止后续处理歧义
    df = _handle_duplicate_columns(df)

    # 4. 强制 Schema 对齐 (白名单机制)
    df = _enforce_schema(df)

    # 5. 类型转换与缺失值填充
    # 排除不需要填充的 Label 列
    fill_cols = [c for c in df.columns if c not in ['Risk_Label', 'Cluster_Label']]
    
    # 转换为数值型 (强制转换，非法值变 NaN)
    df[fill_cols] = df[fill_cols].apply(pd.to_numeric, errors='coerce')
    
    # 填充 NaN
    # 策略：如果是数值列且非全空，用中位数；否则用众数或0
    for col in fill_cols:
        if df[col].isna().any():
            if df[col].notna().sum() > 0:
                fill_val = df[col].median()
            else:
                fill_val = 0.0
            df[col] = df[col].fillna(fill_val)

    print(f"Final shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("-" * 40 + "\n")
    
    return df

if __name__ == "__main__":
    load_and_clean_data()