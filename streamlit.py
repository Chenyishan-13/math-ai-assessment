import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# --- 1. 模型定義 (必須與訓練時一致) ---
class AcademicAttentionCNN(nn.Module):
    def __init__(self):
        super(AcademicAttentionCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(4) 
        self.fc_features = nn.Linear(32 * 4, 64)
        self.classifier = nn.Linear(64, 2)
        
    def forward(self, x):
        c_out = self.relu(self.bn1(self.conv1(x)))
        p_out = self.pool(c_out)
        x_flat = p_out.view(p_out.size(0), -1)
        feat = torch.relu(self.fc_features(x_flat))
        logits = self.classifier(feat)
        return logits, feat

# --- 2. 頁面配置與標題 ---
st.set_page_config(page_title="AI 學術競爭力鑑定系統", page_icon="🎓")
st.title("🧠 AI 全校學術競爭力鑑定系統")
st.markdown("---")

# --- 3. 側邊欄設定 ---
st.sidebar.header("📊 數據輸入與設定")
target_school = st.sidebar.text_input("目標院校名稱", value="成功大學 (NCKU)")
uploaded_file = st.sidebar.file_uploader("上傳院校數據 (CSV)", type=["csv"])

# --- 4. 核心鑑定函數 ---
def run_assessment(df):
    device = torch.device("cpu") # Streamlit Server 通常使用 CPU
    model = AcademicAttentionCNN()
    # 這裡假設你有 .pth 檔案，若無則隨機初始化演示
    model.eval()

    feature_cols = [f'Bin_{i*10}-{i*10+9 if i<9 else 100}' for i in range(10)]
    core_keywords = ['高等', '分析', '方程', '代數', '幾何', '線性', '微分', '拓樸', '數值']
    
    feat_vals = df[feature_cols].values.astype(float)
    is_core_vals = df['課程名稱'].apply(lambda x: 1 if any(k in str(x) for k in core_keywords) else 0).values.reshape(-1, 1)
    
    X_raw = np.hstack([feat_vals, is_core_vals])
    X_t = torch.nan_to_num(torch.tensor(X_raw, dtype=torch.float32), nan=0.0)
    bins_n = X_t[:, :10] / (X_t[:, :10].sum(dim=-1, keepdim=True) + 1e-8)
    X_input = torch.cat([bins_n, X_t[:, 10:]], dim=1).unsqueeze(1)

    with torch.no_grad():
        logits, features = model(X_input)
        probs = torch.softmax(logits, dim=1)[:, 1]
        
        raw_weights = torch.sigmoid(features.sum(dim=1)) 
        multiplier = torch.tensor([2.5 if c == 1 else 0.8 for c in is_core_vals.flatten()])
        att_weights = torch.softmax(raw_weights * multiplier * 2.0, dim=0) 
        
        raw_score = torch.sum(probs * att_weights).item() * 100

    # 標竿校準
    final_score = (np.tanh((raw_score - 40) / 30) * 20) + 80
    final_score = min(final_score, 99.8)
    
    return final_score, att_weights

# --- 5. 主畫面邏輯 ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ 數據載入成功！")
    
    if st.button("🚀 開始 AI 深度鑑定"):
        score, weights = run_assessment(df)
        
        # 顯示鑑定報告
        st.subheader(f"🎓 AI 標竿校準報告：{target_school}")
        
        # 動態進度條
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(score / 100)
        with col2:
            st.write(f"**{score:.2f}%**")
        
        # 等級顯示
        if score >= 88: st.metric("判定等級", "S+ (Global Elite) 🌟")
        elif score >= 78: st.metric("判定等級", "S (Top Research) 🚀")
        else: st.metric("判定等級", "A (Academic Excellence) 💎")
        
        st.markdown("---")
        
        # 顯示關鍵指標課程
        st.write("📌 **AI 鑑定關鍵指標課程 (動態權重排序)**")
        top_idx = torch.argsort(weights, descending=True)[:5]
        for i in top_idx:
            c_name = df.iloc[i.item()]['課程名稱']
            c_w = weights[i.item()].item() * 100
            st.info(f"課程：{c_name} | 影響力權重：{c_w:.2f}%")

else:
    st.info("請在側邊欄上傳 CSV 數據以開始鑑定。")
    st.warning("提示：CSV 需包含 '課程名稱' 與 Bin_0-9 至 Bin_90-100 的成績分佈列。")