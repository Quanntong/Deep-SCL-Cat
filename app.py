import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import time

# ================= 1. 基础配置与路径 =================
# 添加src目录到路径
sys.path.insert(0, os.path.abspath("src"))

try:
    import src.config as config
except ImportError:
    st.error("❌ 无法导入项目配置，请确保你在项目根目录下运行。")
    st.stop()

# 尝试导入绘图库
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import matplotlib.font_manager as fm

# 解决中文乱码 - 使用与 config.py 一致的中文字体配置
# 先设置样式
plt.style.use('default')

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

plt.rcParams['axes.unicode_minus'] = False

# 尝试导入 CatBoost
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# ================= 2. 页面初始化 =================
st.set_page_config(
    page_title="Deep-SCL-Cat 学业预警系统 Pro",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 注入自定义 CSS 样式
st.markdown("""
<style>
    /* 全局中文字体设置 */
    * {
        font-family: 'Microsoft YaHei', 'SimHei', 'SimSun', 'Microsoft JhengHei', 'STXihei', sans-serif !important;
    }
    
    /* 顶部标题样式 */
    .main-header { 
        font-size: 2.2rem; 
        font-weight: 800; 
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem; 
    }
    
    /* 指标卡片样式 */
    .metric-card { 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 12px; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.05); 
        text-align: center; 
        border: 1px solid #f0f0f0; 
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.1);
    }
    .metric-value { 
        font-size: 2.2rem; 
        font-weight: bold; 
        color: #2563EB; 
        margin: 10px 0;
    }
    .metric-label { 
        font-size: 0.95rem; 
        color: #6B7280; 
        font-weight: 600;
        text-transform: uppercase; 
        letter-spacing: 1px;
    }
    .metric-desc {
        font-size: 0.8rem;
        color: #9CA3AF;
    }
    
    /* 风险标签样式 */
    .risk-tag-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .risk-tag-normal {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ================= 3. 资源加载模块 =================
@st.cache_resource
def load_system_resources():
    """
    加载所有必要的模型、预处理器和配置文件
    """
    resources = {
        'clf_model': None, 'reg_model': None, 'scaler': None, 
        'kmeans': None, 'feature_cols': None, 
        'auto_threshold': 0.5, # 默认值
        'loaded': False,
        'status_msg': "初始化..."
    }
    
    if not CATBOOST_AVAILABLE:
        resources['status_msg'] = "❌ 缺少 catboost 库"
        return resources

    try:
        # 定义文件路径
        paths = {
            'clf': os.path.join(config.OUTPUT_DIR, 'catboost_classification.cbm'),
            'reg': os.path.join(config.OUTPUT_DIR, 'catboost_regression.cbm'),
            'scaler': os.path.join(config.OUTPUT_DIR, 'scaler.pkl'),
            'kmeans': os.path.join(config.OUTPUT_DIR, 'kmeans.pkl'),
            'feats': os.path.join(config.OUTPUT_DIR, 'model_feature_cols.pkl'),
            'thresh': os.path.join(config.OUTPUT_DIR, 'best_threshold.txt')
        }
        
        # 检查核心模型是否存在
        if not os.path.exists(paths['clf']): 
            resources['status_msg'] = "⚠️ 模型文件未找到，请先运行 main.py 训练"
            return resources

        # 加载 CatBoost 模型
        clf = CatBoostClassifier()
        clf.load_model(paths['clf'])
        
        reg = CatBoostRegressor()
        reg.load_model(paths['reg'])
        
        # 加载 sklearn 对象
        resources.update({
            'clf_model': clf,
            'reg_model': reg,
            'scaler': joblib.load(paths['scaler']),
            'kmeans': joblib.load(paths['kmeans']),
            'feature_cols': joblib.load(paths['feats']),
            'loaded': True,
            'status_msg': "✅ 系统就绪"
        })
        
        # 加载自动计算的最佳阈值
        if os.path.exists(paths['thresh']):
            with open(paths['thresh'], 'r') as f:
                val = float(f.read().strip())
                resources['auto_threshold'] = val
        
    except Exception as e:
        resources['status_msg'] = f"❌ 加载失败: {str(e)}"
    
    return resources

def align_features(df, required_cols):
    """特征对齐工具"""
    df_aligned = df.copy()
    for col in required_cols:
        if col not in df_aligned.columns:
            df_aligned[col] = 0
    return df_aligned[required_cols]

# ================= 4. 侧边栏与导航 =================
def render_sidebar(resources):
    st.sidebar.markdown("# 🎓 Deep-SCL-Cat")
    st.sidebar.markdown("### 高校学业预警系统 Pro")
    
    # 系统状态指示
    if resources['loaded']:
        st.sidebar.success(resources['status_msg'])
    else:
        st.sidebar.warning(resources['status_msg'])
    
    st.sidebar.markdown("---")
    
    # 导航菜单
    page = st.sidebar.radio(
        "功能导航", 
        ["📊 模型驾驶舱", "🔮 单体风险模拟", "📂 批量智能筛查"],
        index=0
    )
    
    st.sidebar.markdown("---")
    
    # === [核心功能找回] 阈值控制面板 ===
    st.sidebar.markdown("### 🎛️ 判别参数设置")
    
    # 自动阈值展示
    auto_th = resources.get('auto_threshold', 0.5)
    st.sidebar.caption(f"🤖 AI 推荐最佳阈值: **{auto_th:.4f}**")
    
    # 手动干预滑块 (这就是您要的阈值控制)
    user_threshold = st.sidebar.slider(
        "判定阈值 (手动干预)", 
        min_value=0.1, 
        max_value=0.9, 
        value=auto_th, 
        step=0.01,
        help="低于此阈值的概率会被判为正常，高于此阈值判为高危。调低可提高召回率（更严格），调高可减少误报。"
    )
    
    # 将用户设定的阈值存入 resources 供全局使用
    resources['current_threshold'] = user_threshold
    
    if abs(user_threshold - auto_th) > 0.05:
        st.sidebar.info("💡 您正在使用自定义阈值")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("© 2024 Deep-SCL-Cat Team")
    
    return page

# ================= 5. 功能页面：模型驾驶舱 =================
def render_overview(resources):
    st.markdown('<div class="main-header">📊 模型驾驶舱 (Dashboard)</div>', unsafe_allow_html=True)
    
    if not resources['loaded']:
        st.error("请先完成模型训练！")
        return

    # 1. 核心指标卡片
    col1, col2, col3, col4 = st.columns(4)
    
    curr_thresh = resources.get('current_threshold', 0.5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">当前判定阈值</div>
            <div class="metric-value">{curr_thresh:.2f}</div>
            <div class="metric-desc">Risk Threshold</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">高危预警模型</div>
            <div class="metric-value">CatBoost</div>
            <div class="metric-desc">Classifier</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">挂科预测模型</div>
            <div class="metric-value">RMSE</div>
            <div class="metric-desc">Regressor</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">心理画像</div>
            <div class="metric-value">3类</div>
            <div class="metric-desc">K-Means Clustering</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # 2. 可视化图表展示区
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📈 阈值优化曲线 (P-R Curve)")
        opt_img_path = os.path.join(config.OUTPUT_DIR, 'optimal_threshold_curve.png')
        if os.path.exists(opt_img_path):
            st.image(opt_img_path, caption="精确率-召回率权衡分析", use_column_width=True)
        else:
            st.info("暂无优化曲线，请运行 main.py 生成")
            
    with col_right:
        st.subheader("🧬 SHAP 特征重要性")
        img_path = os.path.join(config.OUTPUT_DIR, 'shap_summary_dot.png')
        if os.path.exists(img_path):
            st.image(img_path, caption="SCL-90 因子对风险的影响力排序", use_column_width=True)
        else:
            st.info("SHAP 分析图表暂缺，请运行 src/explainability.py")

    # 3. 文字解读
    with st.expander("📖 查看模型详细解读"):
        st.markdown("""
        **如何理解这些图表？**
        - **左图 (P-R Curve)**：展示了模型在不同阈值下的表现。红点是我们自动计算的最佳平衡点，既能抓住大部分高危学生，又不会产生太多误报。
        - **右图 (SHAP)**：展示了哪些心理因子最致命。
            - 点的**颜色越红**，代表该因子分数越高。
            - 点**越靠右**，代表该因子导致高危的概率越大。
            - 例如：如果"抑郁"因子的红点都在右侧，说明抑郁分越高，挂科风险越大。
        """)

# ================= 6. 功能页面：单体预测 =================
def render_prediction(resources):
    st.markdown('<div class="main-header">🔮 单体学生风险模拟</div>', unsafe_allow_html=True)
    
    if not resources['loaded']:
        st.error("系统未就绪")
        return

    st.markdown("### 1. 输入心理测评数据")
    st.info("请在下方输入该学生的 SCL-90 各项因子得分 (1-5分)")
    
    # 动态生成输入框
    input_data = {}
    cols = st.columns(5)
    features = config.SCL90_FEATURES
    
    for i, feature in enumerate(features):
        with cols[i % 5]:
            input_data[feature] = st.number_input(
                feature, 
                min_value=1.0, 
                max_value=5.0, 
                value=1.5, 
                step=0.1,
                help=config.FACTOR_DEFINITIONS.get(feature, "")
            )
    
    if st.button("🚀 开始评估", type="primary"):
        try:
            # 创建输入数据框
            df_input = pd.DataFrame([input_data])
            
            # 特征缩放和聚类
            X_sc = resources['scaler'].transform(df_input[features])
            df_input['Cluster_Label'] = resources['kmeans'].predict(X_sc)[0]
            
            # 准备模型输入
            X_model = align_features(df_input, resources['feature_cols'])
            
            # 预测
            risk_prob = resources['clf_model'].predict_proba(X_model)[0, 1]
            fail_count = max(0, resources['reg_model'].predict(X_model)[0])
            
            # 使用当前阈值判定
            thresh = resources.get('current_threshold', 0.5)
            is_high_risk = risk_prob > thresh
            
            st.markdown("---")
            
            # 显示结果
            col1, col2 = st.columns(2)
            
            with col1:
                color = "red" if is_high_risk else "green"
                status = "🔴 高危预警" if is_high_risk else "🟢 状态正常"
                st.markdown(f"<h3 style='color:{color}'>{status}</h3>", unsafe_allow_html=True)
                st.metric("风险概率", f"{risk_prob:.1%}", f"阈值: {thresh:.1%}")
                
                if is_high_risk and risk_prob < 0.5:
                    st.caption("⚠️ 注意：该生概率虽未过半，但已触发生命线，建议关注！")
            
            with col2:
                st.metric("预计挂科数", f"{fail_count:.1f} 科")
                
        except Exception as e:
            st.error(f"预测出错: {e}")

# ================= 7. 功能页面：批量筛查 =================
def render_batch(resources):
    st.markdown('<div class="main-header">📂 批量智能筛查</div>', unsafe_allow_html=True)
    
    st.markdown("### 1. 上传数据文件")
    st.info("请上传包含学生SCL-90测评数据的CSV或Excel文件")
    
    uploaded_file = st.file_uploader(
        "选择文件", 
        type=['csv', 'xlsx'],
        help="文件应包含SCL-90因子列：躯体化、强迫症状、人际关系敏感、抑郁、焦虑、敌对、恐怖、偏执、精神病性、其他"
    )
    
    if uploaded_file is not None:
        try:
            # 读取文件
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # 清洗列名
            df.columns = df.columns.astype(str).str.strip().str.replace(r'\s+', '', regex=True)
            
            # 检查必要列
            missing_cols = [col for col in config.SCL90_FEATURES if col not in df.columns]
            if missing_cols:
                st.error(f"文件缺少以下必要列: {missing_cols}")
                return
            
            if st.button("🔍 开始批量分析", type="primary"):
                with st.spinner("正在分析数据..."):
                    # 数据预处理
                    X = df[config.SCL90_FEATURES].fillna(df[config.SCL90_FEATURES].median())
                    
                    # 聚类
                    X_scaled = resources['scaler'].transform(X)
                    df['Cluster_Label'] = resources['kmeans'].predict(X_scaled)
                    
                    # 准备模型输入
                    X_model = df.copy()
                    X_model = align_features(X_model, resources['feature_cols'])
                    
                    # 预测
                    probs = resources['clf_model'].predict_proba(X_model)[:, 1]
                    df['高危概率'] = probs
                    
                    # 使用当前阈值判定
                    thresh = resources.get('current_threshold', 0.5)
                    df['风险标签'] = df['高危概率'].apply(lambda x: '高危' if x > thresh else '正常')
                    
                    # 预测挂科数
                    fail_counts = resources['reg_model'].predict(X_model)
                    df['预计挂科数'] = np.maximum(0, fail_counts)
                    
                    # 显示结果
                    st.success(f"✅ 分析完成！共分析 {len(df)} 名学生")
                    
                    # 统计信息
                    high_risk_count = (df['风险标签'] == '高危').sum()
                    st.metric("高危学生数", f"{high_risk_count} 人", f"占比: {high_risk_count/len(df):.1%}")
                    
                    # 显示数据预览
                    st.subheader("📊 分析结果预览")
                    st.dataframe(df[['高危概率', '风险标签', '预计挂科数'] + config.SCL90_FEATURES[:3]].head(10))
                    
                    # 下载按钮
                    csv = df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="📥 下载完整分析结果",
                        data=csv,
                        file_name="学生风险分析结果.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"分析出错: {e}")

# ================= 8. 主函数 =================
def main():
    """主函数：加载资源并渲染页面"""
    # 加载系统资源
    resources = load_system_resources()
    
    # 渲染侧边栏并获取当前页面
    page = render_sidebar(resources)
    
    # 根据页面选择渲染内容
    if page == "📊 模型驾驶舱":
        render_overview(resources)
    elif page == "🔮 单体风险模拟":
        render_prediction(resources)
    elif page == "📂 批量智能筛查":
        render_batch(resources)

if __name__ == "__main__":
    main()
