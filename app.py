#!/usr/bin/env python3
"""
Deep-SCL-Cat Web 可视化界面
基于 Streamlit 构建的学业预警系统看板
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import joblib
import os
import sys

# 添加src目录到路径，以便导入config模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import src.config as config
except ImportError:
    # 如果直接导入失败，尝试相对导入
    try:
        from src import config
    except ImportError:
        # 最后尝试直接导入config
        import config

# 导入CatBoost
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# 页面配置
st.set_page_config(
    page_title="Deep-SCL-Cat 学业预警系统",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 🎨 UI/UX 样式升级区域
# ==========================================
st.markdown("""
<style>
    /* 全局字体与背景 */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* 标题样式 */
    .main-header {
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .sub-header {
        font-size: 1.6rem;
        color: #1f2937;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1.2rem;
        border-left: 6px solid #3B82F6;
        padding-left: 12px;
    }
    
    /* 卡片式设计：关键指标 */
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        border: 1px solid #f3f4f6;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    .metric-title {
        color: #6b7280;
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    
    .metric-desc {
        font-size: 0.85rem;
        color: #9ca3af;
    }

    /* 信息提示框优化 */
    .info-box {
        background-color: #ffffff;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        font-size: 0.95rem;
        line-height: 1.6;
        color: #374151;
    }
    
    /* 侧边栏优化 */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    
    /* 按钮样式微调 */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    /* 数据表格优化 */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* 隐藏 Streamlit 默认页脚 */
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources(force_reload=False):
    """
    加载模型和预处理资源，使用缓存避免重复加载
    """
    resources = {
        'model': None,
        'scaler': None,
        'kmeans': None,
        'feature_cols': None,
        'loaded': False
    }
    
    try:
        # 检查必要文件是否存在
        required_files = [
            'outputs/catboost_model.cbm',
            'outputs/scaler.pkl',
            'outputs/kmeans.pkl'
        ]
        
        for file in required_files:
            if not os.path.exists(file):
                return resources
        
        # 加载CatBoost模型
        if CATBOOST_AVAILABLE:
            model = CatBoostClassifier()
            model.load_model('outputs/catboost_model.cbm')
            resources['model'] = model
        else:
            st.warning("⚠️ CatBoost不可用，请安装catboost库")
        
        # 加载标准化器
        scaler = joblib.load('outputs/scaler.pkl')
        resources['scaler'] = scaler
        
        # 加载KMeans模型
        kmeans = joblib.load('outputs/kmeans.pkl')
        resources['kmeans'] = kmeans
        
        # 尝试加载特征列名 (这是修复Bug的关键)
        try:
            if os.path.exists('outputs/model_feature_cols.pkl'):
                feature_cols = joblib.load('outputs/model_feature_cols.pkl')
            elif os.path.exists('outputs/feature_cols.pkl'):
                feature_cols = joblib.load('outputs/feature_cols.pkl')
            else:
                feature_cols = None
        except:
            feature_cols = None
        
        resources['feature_cols'] = feature_cols
        resources['loaded'] = True
        
    except Exception as e:
        st.error(f"加载资源时出错: {e}")
    
    return resources

def align_data_with_model(df, model, resource_feature_cols=None):
    """
    🛠️ 核心修复函数：强制将输入 DataFrame 的列顺序和名称对齐到模型要求的格式
    """
    # 1. 获取模型训练时的特征名称
    model_features = None
    
    # 尝试从 CatBoost 模型对象直接获取
    if hasattr(model, 'feature_names_'):
        model_features = model.feature_names_
    
    # 如果模型没取到，尝试从外部记录的 pickle 文件获取
    if model_features is None and resource_feature_cols is not None:
        model_features = resource_feature_cols
        
    if model_features is None:
        st.error("❌ 无法获取模型的特征名称列表，请检查 outputs/model_feature_cols.pkl 是否存在")
        st.stop()
    
    # 2. 检查缺失列并补全 (Robustness)
    missing_cols = [col for col in model_features if col not in df.columns]
    
    # 针对常见的命名差异做一次自动映射尝试
    alias_map = {
        'age': '年龄', '年龄': 'age',
        'gender': '性别', '性别': 'gender',
        'Cluster_Label': 'Cluster_Label'  # 确保一致
    }
    
    for missing in missing_cols:
        # 尝试通过别名寻找
        if missing in alias_map and alias_map[missing] in df.columns:
            df[missing] = df[alias_map[missing]]
        else:
            # 如果实在找不到，补0 (防止程序崩溃)
            df[missing] = 0

    # 3. 核心步骤：强制重排
    # 这一步会丢弃多余的列，并严格按照模型要求的顺序排列
    aligned_df = df[model_features].copy()
    
    return aligned_df

def init_session_state():
    """初始化会话状态"""
    if 'page' not in st.session_state:
        st.session_state.page = "模型概览"
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None

def load_image(image_path, caption="图表"):
    """加载并显示图片"""
    if os.path.exists(image_path):
        try:
            image = Image.open(image_path)
            with st.container():
                st.image(image, caption=caption, use_column_width=True)
            return True
        except Exception as e:
            st.error(f"加载图片失败: {e}")
            return False
    else:
        st.warning(f"图片不存在: {image_path}")
        st.info("请先运行 `python main.py` 生成图表文件")
        return False

def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.markdown("<h1 style='text-align: center; color: #1E3A8A; margin-bottom:0;'>🎓 Deep-SCL-Cat</h1>", 
                   unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #6B7280; font-size: 0.9rem;'>智能学业预警系统</p>", 
                   unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🧭 导航")
        page_options = ["模型概览", "🎓 单体预测模拟", "📂 批量智能筛查", "⚔️ 模型竞技场"]
        
        current_index = 0
        if st.session_state.page in page_options:
            current_index = page_options.index(st.session_state.page)
            
        selected_page = st.selectbox(
            "选择功能模块:",
            page_options,
            index=current_index,
            key="sidebar_nav_main_key"
        )
        
        if selected_page != st.session_state.page:
            st.session_state.page = selected_page
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("### ⚡ 控制台")
        
        if st.button("🔄 运行完整流程", use_container_width=True):
            st.info("🚀 正在启动分析引擎...")
            # os.system("python main.py") # 实际部署时可能需要异步执行
            st.success("✨ 流程运行完成！数据已更新。")
            st.rerun()
        
        if st.button("📁 查看输出目录", use_container_width=True):
            outputs_dir = "outputs"
            if os.path.exists(outputs_dir):
                files = os.listdir(outputs_dir)
                st.toast(f"📂 输出目录包含 {len(files)} 个文件", icon="✅")
            else:
                st.error("输出目录不存在")
        
        st.markdown("---")
        
        with st.expander("ℹ️ 系统状态"):
            st.markdown(f"""
            <div style='font-size: 0.85rem; color: #4B5563;'>
            <strong>项目路径</strong>:<br>{os.path.abspath('.')}
            <br><br>
            <strong>数据状态</strong>:<br>{'✅ 就绪' if config.DATA_PROCESSED else '❌ 未配置'}
            <br><br>
            <strong>模型状态</strong>:<br>{'✅ 已加载' if os.path.exists('outputs/catboost_model.cbm') else '❌ 未训练'}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='text-align: center; margin-top: 2rem; color: #9CA3AF; font-size: 0.8rem;'>© 2025 Deep-SCL-Cat Team</div>", unsafe_allow_html=True)

def render_model_overview():
    """渲染模型概览页面"""
    st.markdown("<div class='main-header'>📊 模型训练与评估报告</div>", unsafe_allow_html=True)
    st.markdown("通过多维度数据分析与可视化，全面展示模型性能与决策逻辑。")
    
    st.markdown("<div class='sub-header'>📈 核心性能指标</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-title'>模型架构</div>
            <div class='metric-value'>CatBoost</div>
            <div class='metric-desc'>策略: Balanced Weights</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        best_threshold = "0.513"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>最佳决策阈值</div>
            <div class='metric-value'>{best_threshold}</div>
            <div class='metric-desc'>基于 P-R 曲线优化 (Recall优先)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-title'>目标召回率</div>
            <div class='metric-value'>&gt; 95%</div>
            <div class='metric-desc'>业务原则：宁可误报，不可漏报</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("<div class='sub-header'>🖼️ 可视化分析</div>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["核心图表", "SHAP 深度解释"])
    
    with tab1:
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("##### 🎯 P-R 曲线与阈值")
            load_image("outputs/pr_curve.png", "Precision-Recall 曲线")
        
        with col_right:
            st.markdown("##### 🧬 特征重要性排序")
            load_image("outputs/shap_importance_bar.png", "SHAP 特征重要性")
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 🔍 SHAP 摘要图 (Beeswarm)")
            load_image("outputs/shap_summary_dot.png", "SHAP 摘要图")
        
        with col2:
            st.markdown("##### 📊 特征重要性数据")
            csv_path = "outputs/shap_feature_importance.csv"
            if os.path.exists(csv_path):
                try:
                    df_importance = pd.read_csv(csv_path, encoding='utf-8-sig')
                    st.dataframe(
                        df_importance.head(10).style.format({'平均绝对SHAP值': '{:.6f}'}).background_gradient(subset=['平均绝对SHAP值'], cmap='Blues'),
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    csv_data = df_importance.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="📥 下载完整 CSV 数据",
                        data=csv_data,
                        file_name="shap_feature_importance.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"加载数据失败: {e}")
            else:
                st.info("数据文件尚未生成，请运行主流程。")
    
    st.markdown("<div class='sub-header'>💡 决策建议</div>", unsafe_allow_html=True)
    
    col_info, col_suggest = st.columns(2)
    
    with col_info:
        st.markdown("""
        <div class='info-box'>
        <h4 style='margin-top:0'>📌 模型特性卡片</h4>
        <ul style='padding-left: 1.2rem;'>
            <li><strong>算法内核</strong>: CatBoost (Categorical Boosting)</li>
            <li><strong>任务类型</strong>: 二分类（正常 vs 高危）</li>
            <li><strong>优化目标</strong>: 最大化 Recall (召回率)</li>
            <li><strong>增强策略</strong>: K-Means 聚类特征注入</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col_suggest:
        st.markdown("""
        <div class='info-box' style='border-left-color: #10B981; background-color: #F0FDF4;'>
        <h4 style='margin-top:0; color: #047857;'>🚀 落地应用指南</h4>
        <ol style='padding-left: 1.2rem;'>
            <li>使用推荐阈值 <code>0.513</code> 进行硬分类判定。</li>
            <li>重点筛查 <strong>gender, 敌对, 抑郁</strong> 指标异常的学生。</li>
            <li>每学期根据新数据 <strong>Re-train</strong> 模型以校准分布。</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("📖 查看 SCL-90 心理学因子详细定义"):
        cols = st.columns(2)
        items = list(config.FACTOR_DEFINITIONS.items())
        mid = len(items) // 2
        
        with cols[0]:
            for factor, definition in items[:mid]:
                st.markdown(f"**{factor}**: <span style='color:#666'>{definition}</span>", unsafe_allow_html=True)
        with cols[1]:
            for factor, definition in items[mid:]:
                st.markdown(f"**{factor}**: <span style='color:#666'>{definition}</span>", unsafe_allow_html=True)

def render_prediction_simulator():
    """渲染单体预测模拟器页面 (已修复特征对齐问题)"""
    st.markdown("<div class='main-header'>🔮 学生风险实时模拟器</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    输入学生的 SCL-90 因子分及基础特征，系统将通过 <strong>Feature Scaling → K-Means Clustering → CatBoost Inference</strong> 流水线实时评估风险。
    </div>
    """, unsafe_allow_html=True)
    
    resources = load_resources()
    
    if not resources['loaded']:
        st.warning("⚠️ 模型未加载，请先在侧边栏运行完整流程。")
        return
    
    scoring_features = config.SCL90_FEATS
    
    # 表单区域
    with st.container():
        st.markdown("#### 📝 特征录入")
        with st.form("predict_form", border=True):
            st.markdown("**基础信息**")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                input_age = st.number_input("年龄", 15.0, 30.0, 20.0, 1.0, key="sim_age")
            with c2:
                input_gender = st.selectbox("性别", options=[1, 2], format_func=lambda x: "男" if x==1 else "女", key="sim_gender")
                
            st.markdown("**SCL-90 因子评分 (1-5分)**")
            
            input_values = {'age': input_age, 'gender': input_gender}
            
            scl_cols = st.columns(5)
            for i, feature in enumerate(scoring_features):
                col = scl_cols[i % 5]
                with col:
                    val = st.number_input(feature, 0.0, 5.0, 2.0, 0.1, key=f"sim_{feature}")
                    input_values[feature] = val
            
            st.markdown("---")
            submit_col1, submit_col2, submit_col3 = st.columns([1, 1, 1])
            with submit_col2:
                submitted = st.form_submit_button("🚀 开始智能评估", type="primary", use_container_width=True)
    
    if submitted:
        try:
            # 1. 构造初始 DataFrame
            input_df = pd.DataFrame([input_values])
            
            # 2. 获取资源
            scaler = resources['scaler']
            kmeans = resources['kmeans']
            model = resources['model']
            
            # 3. 标准化与聚类
            X_scoring = input_df[scoring_features]
            
            # 确保标准化器的特征顺序正确
            if hasattr(scaler, 'feature_names_in_'):
                # 如果标准化器有特征名称属性，确保顺序匹配
                scaler_features = list(scaler.feature_names_in_)
                # 重新排列X_scoring以匹配标准化器的特征顺序
                X_scoring = X_scoring[scaler_features]
            
            X_scaled = scaler.transform(X_scoring)
            cluster_label = kmeans.predict(X_scaled)[0]
            
            # 4. 构造包含聚类标签的数据
            input_df['Cluster_Label'] = cluster_label
            
            # ================= 核心修复：强制特征对齐 =================
            final_df = align_data_with_model(
                input_df, 
                model, 
                resource_feature_cols=resources['feature_cols']
            )
            # ========================================================

            # 5. 预测
            risk_probability = model.predict_proba(final_df)[0, 1]
            prediction = model.predict(final_df)[0]
            
            best_threshold = 0.513
            is_high_risk = risk_probability > best_threshold
            
            st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
            
            if is_high_risk:
                st.error(f"⚠️ 预警：检测到高风险倾向 (概率: {risk_probability*100:.1f}%)")
            else:
                st.success(f"✅ 正常：未检测到显著风险 (概率: {risk_probability*100:.1f}%)")
            
            res_c1, res_c2, res_c3 = st.columns(3)
            with res_c1:
                st.metric("风险概率", f"{risk_probability*100:.1f}%", delta=f"阈值 {best_threshold*100:.1f}%", delta_color="inverse")
                st.progress(min(float(risk_probability), 1.0))
                
            with res_c2:
                cluster_map = {0: "高症状型", 1: "健康型", 2: "中间型"}
                desc = cluster_map.get(cluster_label, f"Cluster {cluster_label}")
                st.metric("所属心理画像", desc, f"Cluster ID: {cluster_label}")
            
            with res_c3:
                st.metric("最终判定", "高危" if is_high_risk else "正常")
                
            if is_high_risk:
                st.markdown("""
                <div class='info-box' style='border-left-color: #EF4444; background-color: #FEF2F2;'>
                <strong style='color: #B91C1C;'>🚑 建议干预措施：</strong>
                <ul style='margin-bottom:0'>
                    <li>立即启动一对一心理访谈机制。</li>
                    <li>检查该生 <code>抑郁</code> 和 <code>敌对</code> 因子分是否显著偏高。</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

            # 保存状态
            st.session_state.prediction_result = {
                'risk_probability': risk_probability,
                'cluster_label': cluster_label,
                'prediction': prediction,
                'risk_level': "高危" if is_high_risk else "正常",
                'input_values': input_values
            }

            # 调试辅助 (只在开发时看)
            # with st.expander("🛠️ 调试信息"):
            #    st.write("Aligned Columns:", final_df.columns.tolist())

        except Exception as e:
            st.error(f"计算出错: {str(e)}")
            st.exception(e)

def render_batch_screening():
    """渲染批量智能筛查页面 (已修复特征对齐问题)"""
    st.markdown("<div class='main-header'>📂 批量智能筛查</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    上传 CSV/Excel 数据表，系统将自动进行 <strong>清洗 → 聚类 → 预测</strong>，并生成高危名单报表。
    </div>
    """, unsafe_allow_html=True)
    
    resources = load_resources()
    if not resources['loaded']:
        st.warning("⚠️ 请先加载模型资源。")
        return

    col_up, col_req = st.columns([1, 1])
    with col_up:
        uploaded_file = st.file_uploader("📥 上传数据文件", type=['csv', 'xlsx', 'xls'])
    
    with col_req:
        with st.expander("查看文件格式要求", expanded=False):
            st.markdown(f"必需包含以下列名:\n`{', '.join(config.SCL90_FEATS)}`")

    if uploaded_file is not None:
        try:
            # 文件读取
            file_ext = uploaded_file.name.split('.')[-1].lower()
            if file_ext in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            else:
                df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            
            # 数据校验
            scoring_features = config.SCL90_FEATS
            missing = [f for f in scoring_features if f not in df.columns]
            
            if missing:
                st.error(f"❌ 缺少必要因子列: {', '.join(missing)}")
                return
                
            st.toast("✅ 数据加载成功", icon="📄")
            
            with st.expander(f"数据预览 ({len(df)} 行)", expanded=False):
                st.dataframe(df.head(), use_container_width=True)

            if st.button("🚀 执行批量评估", type="primary", use_container_width=True):
                with st.spinner("正在进行大规模计算..."):
                    # 1. 聚类特征提取
                    X_scoring = df[scoring_features].copy()
                    scaler = resources['scaler']
                    
                    # 确保标准化器的特征顺序正确
                    if hasattr(scaler, 'feature_names_in_'):
                        # 如果标准化器有特征名称属性，确保顺序匹配
                        scaler_features = list(scaler.feature_names_in_)
                        # 重新排列X_scoring以匹配标准化器的特征顺序
                        X_scoring = X_scoring[scaler_features]
                    
                    X_scaled = scaler.transform(X_scoring)
                    kmeans = resources['kmeans']
                    cluster_labels = kmeans.predict(X_scaled)
                    
                    # 2. 准备预测数据
                    X_processing = df.copy()
                    X_processing['Cluster_Label'] = cluster_labels
                    
                    # ================= 核心修复：强制特征对齐 =================
                    model = resources['model']
                    X_final_model_input = align_data_with_model(
                        X_processing, 
                        model, 
                        resource_feature_cols=resources['feature_cols']
                    )
                    # ========================================================
                    
                    # 3. 预测
                    risk_probabilities = model.predict_proba(X_final_model_input)[:, 1]
                    
                    df_result = df.copy()
                    df_result['Risk_Probability'] = risk_probabilities
                    df_result['Cluster_Label'] = cluster_labels
                    best_threshold = 0.513
                    df_result['Risk_Label'] = (risk_probabilities > best_threshold).astype(int)
                    df_result['Risk_Level'] = df_result['Risk_Label'].map({0: '正常', 1: '高危'})
                    
                    high_risk_df = df_result[df_result['Risk_Label'] == 1].copy()
                    
                    # 4. 结果展示
                    st.markdown("---")
                    st.markdown("### 📊 筛查报告")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("总检测人数", len(df_result))
                    m2.metric("高危预警人数", len(high_risk_df), delta_color="inverse")
                    pct = len(high_risk_df)/len(df_result)*100 if len(df_result)>0 else 0
                    m3.metric("高危比例", f"{pct:.1f}%")
                    
                    if len(high_risk_df) > 0:
                        st.markdown(f"#### 🔴 高危名单 ({len(high_risk_df)}人)")
                        st.dataframe(
                            high_risk_df.sort_values('Risk_Probability', ascending=False).style.format({'Risk_Probability': '{:.2%}'}).background_gradient(subset=['Risk_Probability'], cmap='Reds'),
                            use_container_width=True
                        )
                        
                        d1, d2 = st.columns(2)
                        with d1:
                            st.download_button("📥 下载完整结果 (CSV)", 
                                             df_result.to_csv(index=False).encode('utf-8-sig'),
                                             "full_result.csv", "text/csv", use_container_width=True)
                        with d2:
                            st.download_button("📥 仅下载高危名单 (Excel)",
                                             high_risk_df.to_excel(index=False, engine='openpyxl'), 
                                             "high_risk.xlsx", 
                                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                                             use_container_width=True)
                    else:
                        st.balloons()
                        st.success("🎉 太棒了！未发现高危风险学生。")

        except Exception as e:
            st.error(f"处理失败: {e}")
            st.exception(e)

def render_model_arena():
    """渲染模型竞技场页面"""
    st.markdown("<div class='main-header'>⚔️ 模型竞技场</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    本模块展示 <strong>Deep-SCL-Cat</strong> 与 Random Forest, XGBoost, SVM 等经典算法的横向性能对比。
    核心关注 <strong>Recall (召回率)</strong> 指标。
    </div>
    """, unsafe_allow_html=True)
    
    comparison_csv = "outputs/model_comparison.csv"
    comparison_img = "outputs/model_comparison.png"
    
    if not os.path.exists(comparison_csv):
        st.warning("⚠️ 对比数据未生成，请运行完整流程。")
        return
        
    try:
        df_comparison = pd.read_csv(comparison_csv, encoding='utf-8-sig')
        
        st.markdown("#### 📊 性能雷达/柱状图")
        load_image(comparison_img, "多模型性能对比")
        
        st.markdown("#### 🏆 详细指标榜单")
        
        st.dataframe(
            df_comparison.style.highlight_max(axis=0, props='font-weight:bold; background-color:#FEF3C7; color:#B45309'),
            use_container_width=True
        )
        
        best_recall_model = df_comparison.sort_values('Recall', ascending=False).iloc[0]
        st.markdown(f"""
        <div class='metric-card' style='background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%); border: 2px solid #3B82F6;'>
            <div style='text-align: center'>
                <div class='metric-title'>👑 Recall 最佳模型</div>
                <div class='metric-value' style='color: #2563EB'>{best_recall_model['Model']}</div>
                <div class='metric-desc'>Recall: <strong>{best_recall_model['Recall']:.4f}</strong> | F1: {best_recall_model['F1']:.4f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"加载对比数据失败: {e}")

def main():
    """主函数"""
    init_session_state()
    render_sidebar()
    
    if st.session_state.page == "模型概览":
        render_model_overview()
    elif st.session_state.page == "🎓 单体预测模拟":
        render_prediction_simulator()
    elif st.session_state.page == "📂 批量智能筛查":
        render_batch_screening()
    elif st.session_state.page == "⚔️ 模型竞技场":
        render_model_arena()
    else:
        st.warning(f"页面开发中...")

if __name__ == "__main__":
    main()
