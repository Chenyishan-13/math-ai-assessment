import streamlit as st
import pandas as pd
import torch
import numpy as np
from model import MathDeptCNN

# --- 頁面基本設定 ---
st.set_page_config(page_title="數學系實力 AI 鑑定", page_icon="🎓", layout="centered")

# 自定義 CSS 美化
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎓 數學系所實力 AI 鑑定器")
st.write("透過卷積神經網絡 (CNN) 分析課程成績分布，鑑定學術競爭力等級。")

# --- 1. 載入模型 ---
@st.cache_resource
def load_model():
    model = MathDeptCNN()
    # 確保在 Streamlit Cloud (CPU 環境) 正常運行
    model.load_state_dict(torch.load('math_dept_cnn_v1.pth', map_location='cpu'))
    model.eval()
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"模型載入失敗，請檢查 math_dept_cnn_v1.pth 是否存在。錯誤: {e}")

# --- 2. 檔案上傳 ---
uploaded_file = st.file_uploader("📂 請上傳系所成績分布 CSV 檔案", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # 定義特徵欄位與核心課關鍵字
    feature_cols = [f'Bin_{i*10}-{i*10+9 if i<9 else 100}' for i in range(10)]
    core_keywords = ['微積分', '線性代數', '代數', '分析', '幾何', '拓樸', '微分']
    
    core_probs = []
    found_courses = []

    # --- 3. 執行 AI 鑑定邏輯 ---
    with st.spinner('AI 正在分析數據中...'):
        for _, row in df.iterrows():
            try:
                # 提取成績 Bin 數據
                bins = np.nan_to_num(row[feature_cols].values.astype(float), nan=0.0)
                if bins.sum() == 0: continue
                
                # 正規化並轉換為 Tensor
                input_n = (torch.tensor(bins, dtype=torch.float32) / (bins.sum() + 1e-8)).view(1, 1, 10)
                
                with torch.no_grad():
                    # 預測「頂尖率」機率 (Class 1)
                    prob = torch.softmax(model(input_n), dim=1)[0][1].item()
                
                # 篩選核心課程
                course_name = str(row['課程名稱'])
                if any(key in course_name for key in core_keywords):
                    core_probs.append(prob)
                    found_courses.append(course_name)
            except:
                continue

    # --- 4. 顯示結果報告 ---
    if core_probs:
        score = np.mean(core_probs) * 100
        st.balloons()
        
        st.divider()
        st.subheader("📊 AI 綜合實力報告")
        
        # 評分儀表板
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("核心實力評分", f"{score:.2f}")
        with col2:
            st.write(f"**學力進度條**")
            st.progress(min(score/100, 1.0))

        # 等級判定與診斷建議
        if score >= 50:
            st.success("🏆 **鑑定等級：A (研究型實力強勁)**")
            st.markdown("""
            **診斷報告：**
            * 該系所核心科目展現極高的高分群比例。
            * 學生對抽象理論與嚴謹證明的掌握度符合一線研究型大學水平。
            * 建議持續保持高度競爭力。
            """)
        elif score >= 30:
            st.info("📘 **鑑定等級：B (符合一般學術標準)**")
            st.markdown("""
            **診斷報告：**
            * 學術表現穩定，符合國內一般大學數學系標準。
            * 核心學科仍有提升空間，高分群分布較為稀疏。
            * 建議加強進階選修課的挑戰難度。
            """)
        else:
            st.error("⚠️ **鑑定等級：C (學力表現待加強)**")
            st.markdown(f"""
            **🧪 診斷報告與建議：**
            * **核心實力評分：** {score:.2f} (低於標準線)。
            * **核心課題缺失：** 基礎科目（如微積分、線性代數）的高分群比例偏低，這通常代表學生在基礎概念轉化上有困難。
            * **