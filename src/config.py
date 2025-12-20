import os
from pathlib import Path

# ================= Project Paths =================

# 使用 pathlib 替代 os.path，逻辑更清晰
# .resolve() 解决软链接问题，parent.parent 定位到项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# 保持接口兼容，将 Path 对象转为 str，防止某些老旧库报错
DATA_RAW = str(BASE_DIR / 'data' / 'raw')
DATA_PROCESSED = str(BASE_DIR / 'data' / 'processed')
LOG_DIR = str(BASE_DIR / 'logs')
OUTPUT_DIR = str(BASE_DIR / 'outputs')

# ================= File Configuration =================

RAW_FILE = 'scl90_data.csv' 
PROCESSED_FILE = 'scl90_with_clusters.csv'

# ================= Global Parameters =================

RANDOM_SEED = 42
TEST_SIZE = 0.2

# ================= Domain Constants (SCL-90) =================

SCL90_FACTOR_MAP = {
    'score1': '躯体化', 'score2': '强迫',   'score3': '人际',
    'score4': '抑郁',   'score5': '焦虑',   'score6': '敌对',
    'score7': '恐怖',   'score8': '偏执',   'score9': '精神',
    'score10': '其他'
}

# 特征列表（保持顺序）
SCL90_FEATS = list(SCL90_FACTOR_MAP.values())

FACTOR_DEFINITIONS = {
    '躯体化': '反映心血管、胃肠道、呼吸等系统的躯体不适，如头痛、背痛、肌肉酸痛。',
    '强迫':   '指那些明知没有必要，但无法摆脱的无意义思想、冲动和行为。',
    '人际':   '指不自在感和自卑感，在人际交往中感到自卑、懊丧，心神不安。',
    '抑郁':   '苦闷的情感与心境，生活兴趣减退，动力缺乏，活力丧失等。',
    '焦虑':   '无实质性内容的忐忑不安及神经过敏，包含紧张、烦躁、惊恐。',
    '敌对':   '从思维、情感及行为三方面反映敌对表现，如厌烦、摔物、争论。',
    '恐怖':   '与广场恐怖相关的心理症状，如害怕空旷、人群、社交场合。',
    '偏执':   '主要指投射性思维、敌对、猜疑、关系妄想、被动体验和夸大等。',
    '精神':   '反映精神分裂样症状，如幻听、思维播散、被控制感、思维被插入。',
    '其他':   '主要反映睡眠及饮食情况。'
}

# ================= Utility Functions =================

def make_dirs():
    """Ensure all required project directories exist."""
    dirs_to_create = [DATA_RAW, DATA_PROCESSED, LOG_DIR, OUTPUT_DIR]
    
    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)