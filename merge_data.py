import pandas as pd
import os
import glob
import warnings

# 忽略样式警告
warnings.filterwarnings('ignore')

# === 配置区域 ===
SOURCE_FOLDER = 'SCL-90shuju'  # 原始数据文件夹
TARGET_FILE = os.path.join('data', 'raw', 'scl90_data.csv')

# 定义我们需要保留的“黄金列名” (包含常见别名)
# 脚本会自动把列名里的“因子：躯体化”清洗成“躯体化”
VALID_COLUMNS = [
    '躯体化', '强迫', '人际', '抑郁', '焦虑', '敌对', '恐怖', '偏执', '精神', '饮食', '睡眠',  # 中文名
    'score1', 'score2', 'score3', 'score4', 'score5', 'score6', 'score7', 'score8', 'score9', 'score10', # 英文名
    '学号', '姓名', '性别', '年龄', 'age', 'gender', 'id' # 基础信息
]

def find_header_and_read(file_path):
    """
    智能读取函数：
    1. 寻找包含关键词的真实表头行
    2. 只读取该行以下的数据
    """
    try:
        # 预读前30行找表头
        df_temp = pd.read_excel(file_path, header=None, nrows=30)
        header_idx = -1
        
        for i, row in df_temp.iterrows():
            row_str = row.astype(str).str.cat(sep=' ')
            # 只要包含“躯体化”或“score1”，就认定是表头
            if '躯体化' in row_str or 'score1' in row_str:
                header_idx = i
                break
        
        if header_idx == -1:
            return None, "❌ 未找到含有 SCL-90 因子的表头"

        # 从正确的位置读取
        df = pd.read_excel(file_path, header=header_idx)
        
        # === 核心清洗逻辑 ===
        # 1. 删除全空的行和列
        df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
        
        # 2. 列名清洗：只保留包含关键词的列
        cols_to_keep = []
        for col in df.columns:
            col_str = str(col).strip()
            # 如果列名包含我们要的关键词 (比如 "Unnamed: 1" 肯定不包含)
            # 或者列名本身就是我们要的
            is_valid = False
            for valid_key in VALID_COLUMNS:
                if valid_key in col_str:
                    is_valid = True
                    break
            
            if is_valid:
                cols_to_keep.append(col)
        
        if not cols_to_keep:
            return None, "⚠️ 虽找到表头，但筛选后无有效列 (可能是空表)"
            
        # 只保留筛选后的列
        df_clean = df[cols_to_keep]
        
        return df_clean, f"✅ 成功 (保留 {len(cols_to_keep)} 列)"
        
    except Exception as e:
        return None, f"❌ 读取错误: {str(e)}"

def merge_excel_files():
    print(f"🚀 [终极清洗版] 开始扫描: {SOURCE_FOLDER}")
    
    files = glob.glob(os.path.join(SOURCE_FOLDER, "*.xls*"))
    print(f"📄 发现 {len(files)} 个文件\n")
    
    merged_data = []
    
    for filename in files:
        base_name = os.path.basename(filename)
        df, msg = find_header_and_read(filename)
        
        print(f"   {base_name[:20]:<25} -> {msg}")
        
        if df is not None:
            merged_data.append(df)

    if merged_data:
        print("\n🔄 正在合并...")
        # sort=False 防止列名重排
        final_df = pd.concat(merged_data, ignore_index=True, sort=False)
        
        # === 最后的列名统一 ===
        # 这一步是为了防止 "躯体化" 和 "因子1-躯体化" 分成两列
        # 简单的做法：只要列名里包含 "躯体化"，就重命名为 "躯体化"
        new_columns = {}
        for col in final_df.columns:
            for key in ['躯体化', '强迫症状', '人际敏感', '抑郁', '焦虑', '敌对', '恐怖', '偏执', '精神病性']:
                if key in str(col):
                    new_columns[col] = key # 强制统一中文名
                    break
        
        if new_columns:
            final_df.rename(columns=new_columns, inplace=True)

        print(f" 合并完成！清洗后列数: {len(final_df.columns)}")
        print(f"   最终列名预览: {list(final_df.columns)[:8]} ...")
        
        os.makedirs(os.path.dirname(TARGET_FILE), exist_ok=True)
        final_df.to_csv(TARGET_FILE, index=False, encoding='utf-8-sig')
        print(f"✅ 数据已保存至: {TARGET_FILE}")
    else:
        print("⚠️ 未成功合并任何数据")

if __name__ == "__main__":
    merge_excel_files()