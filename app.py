import streamlit as st
import pandas as pd
import torch
import numpy as np
from model import MathDeptCNN

st.set_page_config(page_title="數學系實力鑑定", page_icon="🎓")
st.title("🎓 數學系所實力 AI 鑑定器")

@st.cache_resource
def load_model():
    model = MathDeptCNN()
    model.load_state_dict(torch.load('math_dept_cnn_v1.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

uploaded_file = st.file_uploader("上傳系所成績 CSV 檔案", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    feature_cols = [f'Bin_{i*10}-{i*10+9 if i<9 else 100}' for i in range(10)]
    core_keywords = ['微積分', '線性代數', '代數', '分析', '幾何', '拓樸', '微分']
    
    core_probs = []
    for _, row in df.iterrows():
        try:
            bins = np.nan_to_num(row[feature_cols].values.astype(float), nan=0.0)
            if bins.sum() == 0: continue
            input_n = (torch.tensor(bins, dtype=torch.float32) / (bins.sum() + 1e-8)).view(1, 1, 10)
            with torch.no_grad():
                prob = torch.softmax(model(input_n), dim=1)[0][1].item()
            if any(key in str(row['課程名稱']) for key in core_keywords):
                core_probs.append(prob)
        except: continue

    if core_probs:
        score = np.mean(core_probs) * 100
        st.balloons()
        st.metric("綜合學力得分", f"{score:.2f}")
        if score >= 50: st.success("鑑定等級：A (頂尖研究水平)")
        else: st.warning("鑑定等級：B/C (一般教學水平)")
    else:
        st.error("找不到核心課程數據，請檢查 CSV 格式。")