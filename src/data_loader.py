import os
import glob
import pandas as pd
import numpy as np
try:
    import src.config as config
except ImportError:
    import config

def standardize_columns(df):
    """
    标准化列名，统一不同年份的表头差异
    """
    # 1. 去除空格和特殊符号
    df.columns = df.columns.astype(str).str.strip().str.replace(r'\s+', '', regex=True)
    
    # 2. 关键列名映射字典
    column_mapping = {
        '人际': '人际关系敏感', '人际敏感': '人际关系敏感',
        '强迫': '强迫症状', 
        '精神': '精神病性', 
        '饮食睡眠': '其他', '饮食': '其他', '睡眠': '其他', 
        '其他(饮食睡眠)': '其他', '因子10': '其他',
        '挂科数': '挂科数目', '挂科': '挂科数目', '挂科数量': '挂科数目',
        '备注（异常因子数）': '备注'
    }
    
    df = df.rename(columns=column_mapping)
    return df

def clean_numeric_column(series):
    """强制转数值，非法字符变NaN"""
    return pd.to_numeric(series, errors='coerce')

def load_and_clean_data():
    print("\n" + "="*60)
    print(">>> [Data Loader] 启动：正在扫描 data/raw/ ...")
    print("="*60)
    
    # 1. 扫描目录下所有csv和xlsx
    raw_path = config.DATA_RAW
    all_files = glob.glob(os.path.join(raw_path, "*.*")) # 扫描所有文件
    
    valid_data_list = []
    
    for filepath in all_files:
        filename = os.path.basename(filepath)
        
        # === 🛡️ 安全过滤机制 ===
        # 只处理文件名包含 "级" 的文件
        if "级" not in filename:
            print(f"⏩ 跳过无关文件: {filename}")
            continue
            
        print(f"📂 正在读取: {filename}")
        
        try:
            # 2. 智能读取逻辑 (兼容伪装成CSV的Excel文件)
            try:
                # 优先尝试作为 Excel 读取 (针对您的特殊情况)
                # 如果真的是CSV，read_excel可能会报错，也可能不仅报错
                if filename.lower().endswith('.xlsx') or filename.lower().endswith('.xls'):
                    df = pd.read_excel(filepath)
                else:
                    # 对于 .csv 后缀，先试着当 CSV 读
                    try:
                        df = pd.read_csv(filepath, encoding='utf-8-sig')
                    except:
                        try:
                            df = pd.read_csv(filepath, encoding='gbk')
                        except:
                            # 关键修改：如果CSV读不出来，尝试用Excel引擎读！
                            print(f"   ⚠️ CSV解码失败，尝试作为Excel格式读取...")
                            df = pd.read_excel(filepath)

            except Exception as e:
                print(f"   ❌ 彻底读取失败: {e}")
                continue

            # 3. 标准化与清洗
            df = standardize_columns(df)
            df['Source_File'] = filename.split('.')[0] # 记录来源
            
            # 4. 检查必要列 (SCL-90)
            existing_features = [c for c in config.SCL90_FEATURES if c in df.columns]
            if len(existing_features) < 5:
                print(f"   ⚠️ 格式不符: 缺少SCL-90核心列，跳过。")
                continue
                
            # 自动补全缺失列
            missing_features = [c for c in config.SCL90_FEATURES if c not in df.columns]
            if missing_features:
                print(f"   🛠️ 自动补全缺失列: {missing_features}")
                for col in missing_features:
                    df[col] = np.nan
            
            # 5. 处理挂科目标列
            if config.TARGET_REGRESSION not in df.columns:
                print(f"   ⚠️ 警告: 未找到'{config.TARGET_REGRESSION}'列，默认设为0")
                df[config.TARGET_REGRESSION] = 0
            else:
                fail_count = (pd.to_numeric(df[config.TARGET_REGRESSION], errors='coerce').fillna(0) > 0).sum()
                print(f"   ✅ 数据有效: 包含 {fail_count} 条挂科记录")

            # 筛选最终列
            keep_cols = ['学号', '姓名', 'Source_File'] + config.SCL90_FEATURES + [config.TARGET_REGRESSION]
            keep_cols = [c for c in keep_cols if c in df.columns]
            
            valid_data_list.append(df[keep_cols])

        except Exception as e:
            print(f"   ❌ 读取发生未知错误: {e}")

    # === 合并与后处理 ===
    if not valid_data_list:
        print("\n❌ 错误: 没有加载到任何有效数据！")
        return None

    full_df = pd.concat(valid_data_list, axis=0, ignore_index=True)
    print("\n>>> 正在合并与清洗全量数据...")

    # 1. 填充 SCL-90 缺失值
    for col in config.SCL90_FEATURES:
        full_df[col] = clean_numeric_column(full_df[col])
        full_df[col] = full_df[col].fillna(full_df[col].median())

    # 2. 清洗挂科数目
    full_df[config.TARGET_REGRESSION] = clean_numeric_column(full_df[config.TARGET_REGRESSION]).fillna(0).astype(int)
    
    # 3. 生成二分类标签
    full_df[config.TARGET_CLASSIFICATION] = (full_df[config.TARGET_REGRESSION] > 0).astype(int)

    print(f"🎉 处理完成!")
    print(f"   总样本量: {len(full_df)}")
    print(f"   总挂科人数: {full_df[config.TARGET_CLASSIFICATION].sum()}")

    # 保存
    config.make_dirs()
    save_path = os.path.join(config.DATA_PROCESSED, config.PROCESSED_FILE)
    full_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"💾 最终数据已保存至: {save_path}")
    
    return full_df

if __name__ == "__main__":
    load_and_clean_data()