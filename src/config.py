import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')

# 文件名配置 (请根据实际文件名修改 RAW_FILE)
RAW_FILE = 'scl90_data.csv' 
PROCESSED_FILE = 'scl90_with_clusters.csv'

# 全局参数
RANDOM_SEED = 42
TEST_SIZE = 0.2

# SCL-90 因子映射
SCL90_FACTOR_MAP = {
    'score1': '躯体化',
    'score2': '强迫症状',
    'score3': '人际敏感',
    'score4': '抑郁',
    'score5': '焦虑',
    'score6': '敌对',
    'score7': '恐怖',
    'score8': '偏执',
    'score9': '精神病性',
    'score10': '其他(睡眠饮食)'
}
# 定义有序的中文特征列表，方便后续调用
SCL90_FEATS = list(SCL90_FACTOR_MAP.values())

# SCL-90 因子详细释义
FACTOR_DEFINITIONS = {
    '躯体化': '反映心血管、胃肠道、呼吸等系统的躯体不适，如头痛、背痛、肌肉酸痛。',
    '强迫症状': '指那些明知没有必要，但无法摆脱的无意义思想、冲动和行为。',
    '人际敏感': '指不自在感和自卑感，在人际交往中感到自卑、懊丧，心神不安。',
    '抑郁': '苦闷的情感与心境，生活兴趣减退，动力缺乏，活力丧失等。',
    '焦虑': '无实质性内容的忐忑不安及神经过敏，包含紧张、烦躁、惊恐。',
    '敌对': '从思维、情感及行为三方面反映敌对表现，如厌烦、摔物、争论。',
    '恐怖': '与广场恐怖相关的心理症状，如害怕空旷、人群、社交场合。',
    '偏执': '主要指投射性思维、敌对、猜疑、关系妄想、被动体验和夸大等。',
    '精神病性': '反映精神分裂样症状，如幻听、思维播散、被控制感、思维被插入。',
    '其他(睡眠饮食)': '主要反映睡眠及饮食情况。'
}

# 创建必要的子目录函数
def make_dirs():
    for d in [DATA_RAW, DATA_PROCESSED, os.path.join(BASE_DIR, 'logs'), os.path.join(BASE_DIR, 'outputs')]:
        os.makedirs(d, exist_ok=True)
