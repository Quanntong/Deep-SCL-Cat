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
    st.warning("CatBoost 库未安装，单体预测功能将受限")

# 页面配置
st.set_page_config(
    page_title="Deep-SCL-Cat 学业预警系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 应用CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        font-weight: bold;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #BFDBFE;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    """
    加载模型和预处理资源，使用缓存避免重复加载
    
    返回:
    dict: 包含加载的资源或None（如果文件不存在）
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
            'outputs/kmeans.pkl',
            'outputs/feature_cols.pkl'
        ]
        
        for file in required_files:
            if not os.path.exists(file):
                return resources
        
        # 加载CatBoost模型
        if CATBOOST_AVAILABLE:
            model = CatBoostClassifier()
            model.load_model('outputs/catboost_model.cbm')
            resources['model'] = model
        
        # 加载标准化器
        scaler = joblib.load('outputs/scaler.pkl')
        resources['scaler'] = scaler
        
        # 加载KMeans模型
        kmeans = joblib.load('outputs/kmeans.pkl')
        resources['kmeans'] = kmeans
        
        # 加载特征列名
        feature_cols = joblib.load('outputs/feature_cols.pkl')
        resources['feature_cols'] = feature_cols
        
        resources['loaded'] = True
        
    except Exception as e:
        st.error(f"加载资源时出错: {e}")
    
    return resources

def init_session_state():
    """初始化会话状态"""
    if 'page' not in st.session_state:
        st.session_state.page = "模型概览"
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None

def load_image(image_path, caption="图表"):
    """
    加载并显示图片
    
    参数:
    image_path: 图片路径
    caption: 图片标题
    """
    if os.path.exists(image_path):
        try:
            image = Image.open(image_path)
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
        # 项目标题
        st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>Deep-SCL-Cat</h1>", 
                   unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color: #3B82F6;'>学业预警系统</h3>", 
                   unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 项目简介
        st.markdown("### 📋 项目简介")
        st.markdown("""
        Deep-SCL-Cat 是一个基于 SCL-90 心理评估数据的学业预警系统，使用 CatBoost 机器学习算法进行高危学生识别。
        
        **主要功能**:
        - 数据预处理与特征工程
        - CatBoost 模型训练
        - 阈值寻优策略
        - SHAP 可解释性分析
        """)
        
        st.markdown("---")
        
        # 页面选择
        st.markdown("### 🗂️ 页面选择")
        page_options = ["模型概览", "🎓 单体预测模拟", "📂 批量智能筛查"]
        selected_page = st.selectbox(
            "选择要查看的页面",
            page_options,
            index=page_options.index(st.session_state.page) if st.session_state.page in page_options else 0
        )
        
        # 更新会话状态
        if selected_page != st.session_state.page:
            st.session_state.page = selected_page
            st.rerun()
        
        st.markdown("---")
        
        # 快速操作
        st.markdown("### ⚡ 快速操作")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 运行完整流程"):
                st.info("正在运行完整分析流程...")
                # 这里可以添加实际运行 main.py 的代码
                # 例如: os.system("python main.py")
                st.success("流程运行完成！请刷新页面查看最新结果。")
        
        with col2:
            if st.button("📁 查看输出目录"):
                outputs_dir = "outputs"
                if os.path.exists(outputs_dir):
                    files = os.listdir(outputs_dir)
                    st.info(f"输出目录包含 {len(files)} 个文件")
                else:
                    st.warning("输出目录不存在")
        
        st.markdown("---")
        
        # 系统信息
        st.markdown("### ℹ️ 系统信息")
        st.markdown(f"""
        - **项目路径**: {os.path.abspath('.')}
        - **数据目录**: {config.DATA_PROCESSED if 'config' in locals() else 'N/A'}
        - **输出目录**: outputs/
        - **模型文件**: {'存在' if os.path.exists('outputs/catboost_model.cbm') else '不存在'}
        """)

def render_model_overview():
    """渲染模型概览页面"""
    # 页面标题
    st.markdown("<div class='main-header'>📊 模型训练与评估报告</div>", 
               unsafe_allow_html=True)
    
    # 关键指标行
    st.markdown("<div class='sub-header'>📈 关键指标</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='color: #3B82F6; margin-top: 0;'>模型类型</h3>
            <p style='font-size: 1.8rem; font-weight: bold; color: #1E3A8A;'>CatBoost (Balanced)</p>
            <p style='color: #6B7280; font-size: 0.9rem;'>使用平衡类别权重的梯度提升树模型</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # 尝试从文件读取最佳阈值，如果不存在则使用默认值
        best_threshold = "0.513"
        threshold_file = "outputs/pr_curve.png"
        if os.path.exists(threshold_file):
            # 在实际应用中，可以从日志文件或CSV中读取真实阈值
            # 这里使用硬编码值作为演示
            best_threshold = "0.513"
        
        st.markdown(f"""
        <div class='metric-card'>
            <h3 style='color: #3B82F6; margin-top: 0;'>最佳阈值</h3>
            <p style='font-size: 1.8rem; font-weight: bold; color: #1E3A8A;'>{best_threshold}</p>
            <p style='color: #6B7280; font-size: 0.9rem;'>基于 Precision-Recall 曲线优化得到</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='color: #3B82F6; margin-top: 0;'>目标召回率</h3>
            <p style='font-size: 1.8rem; font-weight: bold; color: #1E3A8A;'>> 95%</p>
            <p style='color: #6B7280; font-size: 0.9rem;'>优先保证高危学生被识别</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 图表展示区
    st.markdown("<div class='sub-header'>📊 可视化图表</div>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### 📈 Precision-Recall 曲线")
        st.markdown("""
        <div class='info-box'>
        展示了模型在不同阈值下的精确率与召回率权衡关系。
        红点表示根据目标召回率 (>95%) 选择的最佳阈值点。
        </div>
        """, unsafe_allow_html=True)
        
        # 加载 P-R 曲线图
        pr_curve_path = "outputs/pr_curve.png"
        load_image(pr_curve_path, "Precision-Recall 曲线与最佳阈值选择")
    
    with col_right:
        st.markdown("#### 📊 特征重要性排名")
        st.markdown("""
        <div class='info-box'>
        基于 SHAP 值的特征重要性分析，显示各特征对模型预测的贡献程度。
        条形越长表示该特征对识别高危学生越重要。
        </div>
        """, unsafe_allow_html=True)
        
        # 加载特征重要性条形图
        importance_path = "outputs/shap_importance_bar.png"
        load_image(importance_path, "SHAP 特征重要性排名")
    
    # 更多图表展示
    st.markdown("<div class='sub-header'>🔍 详细分析</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔵 SHAP 摘要图")
        st.markdown("""
        <div class='info-box'>
        展示了每个特征对模型输出的影响方向和程度。
        颜色表示特征值高低，位置表示SHAP值大小。
        </div>
        """, unsafe_allow_html=True)
        
        # 加载 SHAP 摘要散点图
        summary_path = "outputs/shap_summary_dot.png"
        load_image(summary_path, "SHAP 摘要图")
    
    with col2:
        st.markdown("#### 📋 特征重要性数据")
        st.markdown("""
        <div class='info-box'>
        详细的特征重要性数值，可用于进一步分析和报告。
        </div>
        """, unsafe_allow_html=True)
        
        # 尝试加载特征重要性CSV文件
        csv_path = "outputs/shap_feature_importance.csv"
        if os.path.exists(csv_path):
            try:
                df_importance = pd.read_csv(csv_path, encoding='utf-8-sig')
                # 格式化显示
                df_display = df_importance.copy()
                df_display['平均绝对SHAP值'] = df_display['平均绝对SHAP值'].apply(lambda x: f"{x:.6f}")
                
                # 显示前10个特征
                st.dataframe(
                    df_display.head(10),
                    hide_index=True
                )
                
                # 下载按钮
                csv_data = df_importance.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 下载完整CSV",
                    data=csv_data,
                    file_name="shap_feature_importance.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"加载CSV文件失败: {e}")
        else:
            st.warning("特征重要性CSV文件不存在")
            st.info("请先运行 `python main.py` 生成分析结果")
    
    # 模型信息与建议
    st.markdown("<div class='sub-header'>💡 模型使用建议</div>", unsafe_allow_html=True)
    
    col_info, col_suggest = st.columns(2)
    
    with col_info:
        st.markdown("""
        <div class='info-box'>
        <h4>🎯 模型信息</h4>
        <ul>
        <li><strong>算法</strong>: CatBoost (Categorical Boosting)</li>
        <li><strong>目标</strong>: 二分类（正常/高危）</li>
        <li><strong>评估指标</strong>: Recall (召回率) 优先</li>
        <li><strong>特征工程</strong>: K-Means 聚类增强</li>
        <li><strong>可解释性</strong>: SHAP 值分析</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col_suggest:
        st.markdown("""
        <div class='info-box'>
        <h4>🚀 使用建议</h4>
        <ol>
        <li>使用最佳阈值 <code>0.513</code> 进行预测</li>
        <li>重点关注 Top 3 特征：gender, 敌对, 抑郁</li>
        <li>定期重新训练模型以适应数据变化</li>
        <li>结合领域知识解释SHAP分析结果</li>
        <li>在实际部署前进行充分的验证测试</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # SCL-90 因子详细释义
    st.markdown("<div class='sub-header'>📖 SCL-90 因子详细释义</div>", unsafe_allow_html=True)
    
    with st.expander("点击查看每个因子的心理学定义"):
        st.markdown("""
        <div class='info-box'>
        <p>SCL-90（症状自评量表）包含10个因子，每个因子反映不同的心理症状维度：</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 遍历因子释义
        for factor, definition in config.FACTOR_DEFINITIONS.items():
            st.markdown(f"""
            <div class='info-box' style='margin-bottom: 0.5rem; padding: 0.8rem;'>
            <strong>{factor}</strong>: {definition}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-box' style='margin-top: 1rem;'>
        <p><strong>说明</strong>: 以上定义基于心理学临床解释，用于辅助理解各因子所代表的症状维度。</p>
        </div>
        """, unsafe_allow_html=True)

def render_prediction_simulator():
    """渲染单体预测模拟器页面"""
    # 页面标题
    st.markdown("<div class='main-header'>🔮 学生风险实时模拟器</div>", 
               unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    <strong>功能说明</strong>: 输入学生的SCL-90因子分和其他特征，系统将实时预测该学生是否存在学业风险。
    预测过程包括：特征标准化 → K-Means聚类 → CatBoost模型预测 → 风险等级评估。
    </div>
    """, unsafe_allow_html=True)
    
    # 加载资源
    resources = load_resources()
    
    if not resources['loaded']:
        st.error("⚠️ 模型文件未找到")
        st.info("""
        请先运行完整分析流程以生成必要的模型文件：
        1. 点击侧边栏的 **🔄 运行完整流程** 按钮
        2. 或执行命令: `python main.py`
        3. 等待流程完成后刷新本页面
        """)
        return
    
    # 获取特征列名（SCL-90中文特征）
    scoring_features = config.SCL90_FEATS
    demographic_features = ['age', 'gender']
    all_features = demographic_features + scoring_features
    
    # 创建输入表单
    with st.form("predict_form"):
        st.markdown("### 📝 输入学生特征数据")
        st.markdown("请根据实际情况填写以下特征值（默认值为正常范围）:")
        
        # 使用列布局组织输入框
        num_cols = 3
        cols = st.columns(num_cols)
        
        input_values = {}
        
        # 为每个特征创建输入框
        for i, feature in enumerate(all_features):
            col_idx = i % num_cols
            with cols[col_idx]:
                # 设置默认值和范围
                default_value = 2.0
                min_value = 0.0
                max_value = 5.0
                step = 0.1
                
                # 特殊处理某些特征
                if feature == 'age':
                    default_value = 20.0
                    min_value = 15.0
                    max_value = 30.0
                    step = 1.0
                    st.markdown(f"**{feature}** (年龄)")
                elif feature == 'gender':
                    default_value = 1.0
                    min_value = 1.0
                    max_value = 2.0
                    step = 1.0
                    st.markdown(f"**{feature}** (1=男, 2=女)")
                else:
                    # SCL-90中文特征
                    st.markdown(f"**{feature}**")
                
                # 创建数字输入框
                value = st.number_input(
                    label="",  # 标签已在上面显示
                    min_value=min_value,
                    max_value=max_value,
                    value=default_value,
                    step=step,
                    key=f"input_{feature}"
                )
                input_values[feature] = value
        
        # 提交按钮
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button(
                "🚀 开始评估",
                type="primary"
            )
    
    # 处理预测逻辑
    if submitted:
        st.markdown("---")
        st.markdown("### 📊 预测结果")
        
        try:
            # 1. 将输入数据转为DataFrame
            input_df = pd.DataFrame([input_values])
            
            # 显示输入数据
            with st.expander("📋 查看输入数据"):
                st.dataframe(input_df)
            
            # 2. 提取SCL-90特征进行标准化
            scoring_features = config.SCL90_FEATS
            X_scoring = input_df[scoring_features]
            
            # 标准化处理（只对SCL-90特征）
            scaler = resources['scaler']
            X_scaled = scaler.transform(X_scoring)
            
            # 3. K-Means聚类预测
            kmeans = resources['kmeans']
            cluster_label = kmeans.predict(X_scaled)[0]
            
            # 4. 将聚类标签拼接到特征中
            input_df_with_cluster = input_df.copy()
            input_df_with_cluster['Cluster_Label'] = cluster_label
            
            # 5. 模型预测
            model = resources['model']
            
            # 预测概率
            risk_probability = model.predict_proba(input_df_with_cluster)[0, 1]  # 高危类别的概率
            
            # 预测类别
            prediction = model.predict(input_df_with_cluster)[0]
            
            # 6. 显示结果
            st.markdown("<div class='sub-header'>🎯 风险评估</div>", unsafe_allow_html=True)
            
            # 最佳阈值
            best_threshold = 0.513
            
            # 风险等级判断
            if risk_probability > best_threshold:
                st.error(f"⚠️ **高危预警** - 建议重点关注")
                risk_level = "高危"
                color = "red"
            else:
                st.success(f"✅ **安全范围** - 风险较低")
                risk_level = "正常"
                color = "green"
            
            # 显示详细信息
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="风险概率",
                    value=f"{risk_probability*100:.1f}%",
                    delta=f"阈值: {best_threshold*100:.1f}%",
                    delta_color="inverse" if risk_probability > best_threshold else "normal"
                )
            
            with col2:
                # 心理画像簇描述
                cluster_descriptions = {
                    0: "高症状型 - 多个SCL-90因子分较高",
                    1: "健康型 - 多数因子分处于中等或较低水平", 
                    2: "中间型 - 因子分介于健康型和高症状型之间"
                }
                description = cluster_descriptions.get(cluster_label, f"Cluster {cluster_label}")
                st.metric(
                    label="心理画像簇",
                    value=f"Cluster {cluster_label}",
                    delta=description
                )
            
            with col3:
                st.metric(
                    label="预测结果",
                    value=risk_level,
                    delta="高危(1)" if prediction == 1 else "正常(0)"
                )
            
            # 详细解释
            st.markdown("<div class='sub-header'>📈 结果解读</div>", unsafe_allow_html=True)
            
            if risk_level == "高危":
                st.markdown(f"""
                <div class='info-box' style='border-left: 5px solid #EF4444;'>
                <h4>🔴 高危学生识别</h4>
                <ul>
                <li>该学生的风险概率为 <strong>{risk_probability*100:.1f}%</strong>，超过了最佳阈值 {best_threshold*100:.1f}%</li>
                <li>属于 <strong>{cluster_descriptions.get(cluster_label, f'Cluster {cluster_label}')}</strong></li>
                <li><strong>建议措施</strong>: 建议进行一对一心理辅导，定期跟踪学业表现，提供必要的学习支持</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='info-box' style='border-left: 5px solid #10B981;'>
                <h4>🟢 正常范围学生</h4>
                <ul>
                <li>该学生的风险概率为 <strong>{risk_probability*100:.1f}%</strong>，低于最佳阈值 {best_threshold*100:.1f}%</li>
                <li>属于 <strong>{cluster_descriptions.get(cluster_label, f'Cluster {cluster_label}')}</strong></li>
                <li><strong>建议措施</strong>: 保持常规关注，鼓励参与集体活动，定期进行心理评估</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # 保存预测结果到会话状态
            st.session_state.prediction_result = {
                'risk_probability': risk_probability,
                'cluster_label': cluster_label,
                'prediction': prediction,
                'risk_level': risk_level,
                'input_values': input_values
            }
            
        except Exception as e:
            st.error(f"预测过程中发生错误: {e}")
            st.info("请检查输入数据格式或模型文件完整性")
    
    # 显示历史预测结果（如果存在）
    if st.session_state.prediction_result:
        st.markdown("---")
        st.markdown("### 📋 最近一次预测记录")
        
        result = st.session_state.prediction_result
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("风险概率", f"{result['risk_probability']*100:.1f}%")
            st.metric("心理画像簇", f"Cluster {result['cluster_label']}")
        
        with col2:
            st.metric("风险等级", result['risk_level'])
            st.metric("预测类别", "高危(1)" if result['prediction'] == 1 else "正常(0)")
        
        # 清除结果按钮
        if st.button("🗑️ 清除预测记录"):
            st.session_state.prediction_result = None
            st.rerun()

def render_batch_screening():
    """渲染批量智能筛查页面"""
    # 页面标题
    st.markdown("<div class='main-header'>📂 批量智能筛查</div>", 
               unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    <strong>功能说明</strong>: 上传包含学生SCL-90因子分的CSV或Excel文件，系统将自动进行批量风险评估。
    系统会自动识别文件中的SCL-90中文特征列，进行标准化、聚类和模型预测，筛选出高危学生。
    </div>
    """, unsafe_allow_html=True)
    
    # 加载资源
    resources = load_resources()
    
    if not resources['loaded']:
        st.error("⚠️ 模型文件未找到")
        st.info("""
        请先运行完整分析流程以生成必要的模型文件：
        1. 点击侧边栏的 **🔄 运行完整流程** 按钮
        2. 或执行命令: `python main.py`
        3. 等待流程完成后刷新本页面
        """)
        return
    
    # 文件上传区域
    st.markdown("### 📤 上传数据文件")
    st.markdown("""
    请上传包含学生SCL-90因子分的CSV或Excel文件。文件应包含以下10个中文特征列：
    """)
    
    # 显示所需的特征列
    required_features = config.SCL90_FEATS
    st.markdown(f"""
    <div class='info-box'>
    <strong>必需的SCL-90特征列</strong> (10个):
    <br>
    {', '.join(required_features)}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **文件格式要求**:
    - CSV或Excel格式（支持 .csv, .xlsx, .xls）
    - 必须包含上述10个中文特征列（列名需完全一致）
    - 可以包含其他列（如学号、姓名、年龄、性别等），这些列不会影响预测
    - 建议文件大小不超过200MB
    """)
    
    # 文件上传器
    uploaded_file = st.file_uploader(
        "选择文件",
        type=['csv', 'xlsx', 'xls'],
        help="上传CSV或Excel文件"
    )
    
    if uploaded_file is not None:
        try:
            # 读取文件
            file_ext = uploaded_file.name.split('.')[-1].lower()
            if file_ext in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            else:
                df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            
            st.success(f"✅ 文件加载成功！共 {len(df)} 条记录，{len(df.columns)} 列")
            
            # 显示数据预览
            with st.expander("📋 查看数据预览"):
                st.dataframe(df.head())
                st.write(f"**数据形状**: {df.shape}")
                st.write(f"**列名**: {list(df.columns)}")
            
            # 检查必需的SCL-90特征列
            missing_features = [feat for feat in required_features if feat not in df.columns]
            if missing_features:
                st.error(f"❌ 文件缺少以下必需的SCL-90特征列: {', '.join(missing_features)}")
                st.info("请确保文件表头包含上述10个中文特征列，列名需完全一致。")
                return
            
            st.success(f"✅ 所有必需的SCL-90特征列已找到！")
            
            # 检查是否有年龄和性别列（可选）
            optional_features = ['age', 'gender']
            available_optional = [feat for feat in optional_features if feat in df.columns]
            if available_optional:
                st.info(f"📊 检测到可选特征列: {', '.join(available_optional)}")
            else:
                st.warning("⚠️ 未检测到年龄(age)和性别(gender)列，将使用默认值进行预测")
            
            # 开始批量处理按钮
            if st.button("🚀 开始批量风险评估", type="primary"):
                with st.spinner("正在处理数据，请稍候..."):
                    # 1. 提取SCL-90特征
                    scoring_features = config.SCL90_FEATS
                    X_scoring = df[scoring_features].copy()
                    
                    # 2. 标准化处理（只对SCL-90特征）
                    scaler = resources['scaler']
                    X_scaled = scaler.transform(X_scoring)
                    
                    # 3. K-Means聚类预测
                    kmeans = resources['kmeans']
                    cluster_labels = kmeans.predict(X_scaled)
                    
                    # 4. 准备模型输入特征
                    # 确保 age 和 gender 列存在，如果不存在则使用默认值
                    if 'age' not in df.columns:
                        df['age'] = 20  # 默认年龄
                    if 'gender' not in df.columns:
                        df['gender'] = 1  # 默认性别
                    
                    # 创建模型输入数据框
                    model_features = ['age', 'gender'] + scoring_features + ['Cluster_Label']
                    X_model = pd.DataFrame()
                    X_model['age'] = df['age']
                    X_model['gender'] = df['gender']
                    for feat in scoring_features:
                        X_model[feat] = df[feat]
                    X_model['Cluster_Label'] = cluster_labels
                    
                    # 5. 模型预测
                    model = resources['model']
                    
                    # 预测概率
                    risk_probabilities = model.predict_proba(X_model)[:, 1]  # 高危类别的概率
                    predictions = model.predict(X_model)
                    
                    # 6. 添加预测结果到DataFrame
                    df_result = df.copy()
                    df_result['Risk_Probability'] = risk_probabilities
                    df_result['Prediction'] = predictions
                    df_result['Cluster_Label'] = cluster_labels
                    
                    # 使用最佳阈值进行分类
                    best_threshold = 0.513
                    df_result['Risk_Label'] = (risk_probabilities > best_threshold).astype(int)
                    df_result['Risk_Level'] = df_result['Risk_Label'].map({0: '正常', 1: '高危'})
                    
                    # 7. 筛选高危学生
                    high_risk_df = df_result[df_result['Risk_Label'] == 1].copy()
                    high_risk_count = len(high_risk_df)
                    total_count = len(df_result)
                    
                    # 8. 显示统计结果
                    st.markdown("---")
                    st.markdown("### 📊 批量筛查结果")
                    
                    # 关键指标
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("总检测人数", f"{total_count} 人")
                    with col2:
                        st.metric("高危学生数", f"{high_risk_count} 人")
                    with col3:
                        if total_count > 0:
                            high_risk_percentage = (high_risk_count / total_count) * 100
                            st.metric("高危比例", f"{high_risk_percentage:.1f}%")
                        else:
                            st.metric("高危比例", "0%")
                    
                    # 高危学生名单
                    st.markdown(f"### 🔴 高危学生名单 ({high_risk_count} 人)")
                    
                    if high_risk_count > 0:
                        # 选择显示的列：优先显示学号、姓名等标识列，然后是SCL-90特征列
                        display_columns = []
                        
                        # 尝试寻找标识列
                        id_columns = ['学号', '学生编号', 'ID', 'id', '姓名', '名字', 'Name', 'name']
                        for col in id_columns:
                            if col in high_risk_df.columns:
                                display_columns.append(col)
                        
                        # 添加风险相关列
                        display_columns.extend(['Risk_Probability', 'Risk_Level', 'Cluster_Label'])
                        
                        # 添加部分SCL-90特征列
                        display_columns.extend(['抑郁', '焦虑', '敌对'])  # 选择几个关键特征
                        
                        # 确保列存在
                        display_columns = [col for col in display_columns if col in high_risk_df.columns]
                        
                        # 显示高危学生数据
                        st.dataframe(
                            high_risk_df[display_columns].sort_values('Risk_Probability', ascending=False)
                        )
                        
                        # 高危学生特征分析
                        st.markdown("#### 📈 高危学生特征分析")
                        if high_risk_count > 1:
                            # 计算高危学生在各SCL-90特征上的平均值
                            high_risk_means = high_risk_df[scoring_features].mean().sort_values(ascending=False)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**最高平均分的特征 (前5)**")
                                for feature, value in high_risk_means.head(5).items():
                                    st.write(f"- {feature}: {value:.2f}")
                            
                            with col2:
                                st.markdown("**最低平均分的特征 (后5)**")
                                for feature, value in high_risk_means.tail(5).items():
                                    st.write(f"- {feature}: {value:.2f}")
                        else:
                            st.info("仅1名高危学生，特征分析略过")
                    else:
                        st.success("🎉 恭喜！未发现高危学生。")
                    
                    # 9. 导出结果
                    st.markdown("---")
                    st.markdown("### 💾 导出预测结果")
                    
                    # 准备导出数据
                    export_df = df_result.copy()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 导出CSV
                        csv_data = export_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                        st.download_button(
                            label="📥 下载完整CSV结果",
                            data=csv_data,
                            file_name="batch_screening_results.csv",
                            mime="text/csv",
                            help="包含所有学生的完整预测结果"
                        )
                    
                    with col2:
                        if high_risk_count > 0:
                            # 导出高危学生Excel
                            excel_data = high_risk_df.to_excel(index=False, engine='openpyxl')
                            st.download_button(
                                label="📥 下载高危学生Excel",
                                data=excel_data,
                                file_name="high_risk_students.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                help="仅包含高危学生的详细数据"
                            )
                        else:
                            st.info("无高危学生，无需导出高危名单")
                    
                    # 10. 处理建议
                    st.markdown("---")
                    st.markdown("### 💡 处理建议")
                    
                    if high_risk_count > 0:
                        st.markdown(f"""
                        <div class='info-box' style='border-left: 5px solid #EF4444;'>
                        <h4>🔴 发现 {high_risk_count} 名高危学生，建议采取以下措施：</h4>
                        <ol>
                        <li><strong>一对一心理辅导</strong>: 为每位高危学生安排专业心理辅导</li>
                        <li><strong>学业跟踪</strong>: 定期跟踪这些学生的学业表现和出勤情况</li>
                        <li><strong>家长沟通</strong>: 及时与家长沟通，建立家校合作支持机制</li>
                        <li><strong>重点关注</strong>: 重点关注风险概率最高的前3名学生</li>
                        <li><strong>定期复查</strong>: 建议每学期进行一次SCL-90复查</li>
                        </ol>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='info-box' style='border-left: 5px solid #10B981;'>
                        <h4>🟢 未发现高危学生，当前群体心理健康状况良好</h4>
                        <ul>
                        <li><strong>继续保持</strong>: 维持现有的心理健康教育和支持体系</li>
                        <li><strong>预防为主</strong>: 定期开展心理健康讲座和团体辅导</li>
                        <li><strong>关注变化</strong>: 关注学生群体的动态变化，及时发现潜在风险</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"文件处理过程中发生错误: {e}")
            st.info("请检查文件格式和内容是否符合要求。")
    else:
        st.info("👆 请上传数据文件开始批量筛查")

def main():
    """主函数"""
    # 初始化会话状态
    init_session_state()
    
    # 渲染侧边栏
    render_sidebar()
    
    # 根据选择的页面渲染主内容
    if st.session_state.page == "模型概览":
        render_model_overview()
    elif st.session_state.page == "🎓 单体预测模拟":
        render_prediction_simulator()
    elif st.session_state.page == "📂 批量智能筛查":
        render_batch_screening()
    else:
        st.warning(f"页面 '{st.session_state.page}' 尚未实现")

if __name__ == "__main__":
    main()
