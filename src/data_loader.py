import pandas as pd
import numpy as np
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


def load_and_clean_data(filepath=None):
    """
    加载并清洗数据
    
    参数:
    filepath (str, optional): CSV文件路径。如果为None，则使用config.RAW_FILE作为默认路径
    
    返回:
    pd.DataFrame: 清洗后的DataFrame
    """
    # 如果未提供文件路径，使用默认路径
    if filepath is None:
        # 构建完整文件路径：data/raw目录下的RAW_FILE
        filepath = os.path.join(config.DATA_RAW, config.RAW_FILE)
    
    print(f"正在加载数据: {filepath}")
    
    try:
        # 读取CSV文件
        df = pd.read_csv(filepath)
        print(f"原始数据形状: {df.shape}")
    except FileNotFoundError:
        print(f"错误: 文件未找到 - {filepath}")
        print("请确保文件存在，或检查config.py中的RAW_FILE配置")
        return None
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None
    
    # 1. 删除重复行
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed_duplicates = initial_rows - len(df)
    print(f"删除重复行: {removed_duplicates} 行")
    
    # 2. 检查缺失值
    missing_counts = df.isnull().sum()
    missing_percentage = (missing_counts / len(df)) * 100
    
    print("\n缺失值统计:")
    for col in df.columns:
        if missing_counts[col] > 0:
            print(f"  {col}: {missing_counts[col]} 个缺失值 ({missing_percentage[col]:.2f}%)")
    
    # 处理缺失值
    for col in df.columns:
        if missing_counts[col] > 0:
            missing_pct = missing_percentage[col]
            
            if missing_pct < 5:
                # 缺失比例小于5%，删除含有缺失值的行
                df = df.dropna(subset=[col])
                print(f"  {col}: 缺失比例 {missing_pct:.2f}% < 5%，已删除缺失行")
            else:
                # 缺失比例大于等于5%，用中位数填充
                if pd.api.types.is_numeric_dtype(df[col]):
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    print(f"  {col}: 缺失比例 {missing_pct:.2f}% >= 5%，已用中位数 {median_val:.2f} 填充")
                else:
                    # 对于非数值列，用众数填充
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else "未知"
                    df[col] = df[col].fillna(mode_val)
                    print(f"  {col}: 缺失比例 {missing_pct:.2f}% >= 5%，已用众数 '{mode_val}' 填充")
    
    # 3. 确保所有特征列为numeric类型
    # 这里假设除了目标变量外的所有列都是特征
    # 在实际应用中，可能需要根据具体数据调整
    numeric_converted = 0
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                # 尝试转换为数值类型
                df[col] = pd.to_numeric(df[col], errors='coerce')
                numeric_converted += 1
            except:
                # 如果转换失败，保留原类型
                print(f"  警告: 列 '{col}' 无法转换为数值类型，将保持原类型")
    
    if numeric_converted > 0:
        print(f"已将 {numeric_converted} 列转换为数值类型")
    
    # 再次检查是否还有缺失值（由于类型转换可能产生新的缺失值）
    df = df.dropna()
    
    print(f"\n清洗后数据形状: {df.shape}")
    print(f"数据清洗完成")
    
    # 重命名 SCL-90 因子列
    df.rename(columns=config.SCL90_FACTOR_MAP, inplace=True)
    
    return df


if __name__ == "__main__":
    # 主程序入口
    print("=" * 50)
    print("数据加载与清洗模块")
    print("=" * 50)
    
    # 调用数据加载函数
    cleaned_data = load_and_clean_data()
    
    if cleaned_data is not None:
        print("\n" + "=" * 50)
        print("数据加载成功!")
        print(f"数据集形状: {cleaned_data.shape}")
        
        # 检查是否还有NaN
        remaining_nans = cleaned_data.isnull().sum().sum()
        if remaining_nans == 0:
            print("缺失值检查: 数据中已无NaN值")
        else:
            print(f"缺失值检查: 数据中仍有 {remaining_nans} 个NaN值")
        
        # 显示数据前几行
        print("\n数据前5行:")
        print(cleaned_data.head())
        
        # 显示数据类型
        print("\n数据类型:")
        print(cleaned_data.dtypes)
    else:
        print("数据加载失败，请检查错误信息")
