# src/config.py
import os
from pathlib import Path
import matplotlib.pyplot as plt

# ================= 绘图字体配置 (解决中文方框问题) =================
# 更全面的中文字体配置，优先使用系统中可用的字体
import matplotlib
import matplotlib.font_manager as fm

# 添加中文字体路径到字体管理器
# 获取系统中可用的中文字体
chinese_fonts = []
for font in fm.fontManager.ttflist:
    font_name = font.name.lower()
    if 'yahei' in font_name or 'simhei' in font_name or 'simsun' in font_name or 'microsoft jhenghei' in font_name:
        chinese_fonts.append(font.name)

# 设置字体配置
if chinese_fonts:
    matplotlib.rcParams['font.sans-serif'] = chinese_fonts + ['DejaVu Sans', 'Arial Unicode MS']
else:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans', 'Arial Unicode MS']

plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像是负号'-'显示为方块的问题

# ================= 路径配置 =================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = str(BASE_DIR / 'data' / 'raw')
DATA_PROCESSED = str(BASE_DIR / 'data' / 'processed')
LOG_DIR = str(BASE_DIR / 'logs')
OUTPUT_DIR = str(BASE_DIR / 'outputs')

# ================= 文件配置 =================
# 自动扫描 raw 目录下的所有相关csv文件
RAW_FILES_PATTERN = "*.csv" 
PROCESSED_FILE = 'merged_scl90_data.csv'

# ================= 全局参数 =================
RANDOM_SEED = 42
TEST_SIZE = 0.2

# ================= 特征工程配置 =================
# 1. 核心特征 (SCL-90因子)：所有年级都有的公共列
# 注意：我们直接使用中文列名，因为新数据已经是中文了
SCL90_FEATURES = [
    '躯体化', '强迫症状', '人际关系敏感', '抑郁', '焦虑', 
    '敌对', '恐怖', '偏执', '精神病性', '其他'
]

# 2. 辅助特征 (如果有的话，比如性别、年龄，但在你的CSV里需要提取或清洗)
# 暂时只用SCL-90以保证最大兼容性
BASE_FEATURES = [] 

# 3. 目标变量
TARGET_CLASSIFICATION = 'Is_High_Risk' # 二分类目标：是否挂科 (挂科数 > 0)
TARGET_REGRESSION = '挂科数目'         # 回归目标：具体挂几科

# ================= 业务规则 =================
# 定义 SCL-90 因子含义（用于可视化说明）
FACTOR_DEFINITIONS = {
    '躯体化': '反映身体不适感，如头痛、肌肉酸痛等',
    '强迫症状': '指无法摆脱的无意义思想或冲动',
    '人际关系敏感': '指人际交往中的不自在与自卑感',
    '抑郁': '苦闷、生活兴趣减退、动力缺乏',
    '焦虑': '无实质内容的紧张、烦躁与惊恐',
    '敌对': '厌烦、争论、摔物等攻击性行为',
    '恐怖': '对特定场所或事物的非理性恐惧',
    '偏执': '猜疑、敌对、关系妄想',
    '精神病性': '幻听、思维播散等精神分裂样症状',
    '其他': '睡眠及饮食情况'
}

def make_dirs():
    """确保所有目录存在"""
    for d in [DATA_RAW, DATA_PROCESSED, LOG_DIR, OUTPUT_DIR]:
        os.makedirs(d, exist_ok=True)
