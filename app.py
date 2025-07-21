import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ===== 頁面設定 =====
st.set_page_config(page_title="🎲 AI 百家樂 ML 自動化預測下注系統 🎲", page_icon="🎰", layout="centered")

# ===== 激活碼與管理員 =====
ADMIN_PASSWORD = "admin999"
PASSWORDS = {"aa17888", "bb12345"}

if "access_granted" not in st.session_state:
    st.session_state.access_granted = False
    st.session_state.is_admin = False

if not st.session_state.access_granted:
    password_input = st.text_input("輸入激活碼或管理員密碼", type="password")
    if st.button("確認"):
        if password_input in PASSWORDS:
            st.session_state.access_granted = True
        elif password_input == ADMIN_PASSWORD:
            st.session_state.access_granted = True
            st.session_state.is_admin = True
        else:
            st.error("密碼錯誤")
    st.stop()

# ===== 資料庫連線 =====
conn = sqlite3.connect("baccarat.db", check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    result TEXT,
    predict TEXT,
    confidence REAL,
    profit INTEGER,
    created TIMESTAMP
)''')
conn.commit()

# ===== Session 狀態 =====
for k, v in {"history": [], "profit": 0, "wins": 0, "total": 0, "base_bet": 100, "current_bet": 100, "auto_bet": False, "max_loss": -1000}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ===== 模型訓練 =====
df = pd.read_sql_query("SELECT * FROM records ORDER BY created ASC", conn)
df = df[df['result'].isin(['B','P','T'])].copy()
df['result_code'] = df['result'].map({'B':1, 'P':0, 'T':2})
N = 5
features, labels = [], []
for i in range(len(df)-N):
    features.append(df['result_code'].iloc[i:i+N])
    labels.append(df['result_code'].iloc[i+N])

X, y = np.array(features), np.array(labels)
model, accuracy, can_predict = None, None, False

if len(X) >= 15:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    can_predict = True
else:
    st.warning("資料不足無法訓練模型，請先輸入至少 15 筆資料")

# ===== 預測函數 =====
def ml_predict():
    if not can_predict or len(st.session_state.history) < N:
        return "觀望", 0.0
    mapping = {'B': 1, 'P': 0, 'T': 2}
    recent = [mapping.get(x,0) for x in st.session_state.history[-N:]]
    pred = model.predict([recent])[0]
    prob = max(model.predict_proba([recent])[0])
    return ("莊" if pred==1 else "閒" if pred==0 else "和"), prob

# ===== 自動下注邏輯 =====
def auto_bet(pred, prob, threshold=0.65):
    if prob < threshold:
        return "信心不足，觀望"
    if st.session_state.profit <= st.session_state.max_loss:
        st.warning("已達當日最大虧損限制，停止自動下注")
        st.session_state.auto_bet = False
        return "停止下注"
    st.session_state.history.append('B' if pred=="莊" else 'P' if pred=="閒" else 'T')
    c.execute("INSERT INTO records (result, predict, confidence, profit, created) VALUES (?, ?, ?, ?, ?)",
              (pred[0], pred, prob, 0, datetime.datetime.now()))
    conn.commit()
    return f"已自動下注：{pred}"

# ===== 介面 =====
st.title("🎲 AI 百家樂 ML 自動化預測下注系統 🎲")
if can_predict:
    pred_label, pred_prob = ml_predict()
    st.subheader(f"預測結果：{pred_label} (信心 {pred_prob:.2f})")
    st.write(f"模型準確度：{accuracy:.2%}")
else:
    st.info("尚無法預測")

# ===== 自動下注控制 =====
st.checkbox("啟用自動下注 (信心大於 0.65 自動下注)", key="auto_bet")
max_loss_input = st.number_input("設定當日最大虧損限制 (元)", value=st.session_state.max_loss)
st.session_state.max_loss = max_loss_input
if st.session_state.auto_bet and can_predict:
    st.success(auto_bet(pred_label, pred_prob))

# ===== 管理員後台 =====
if st.session_state.is_admin:
    st.header("🛠️ 管理員後台")
    if st.button("清空所有資料"):
        c.execute("DELETE FROM records")
        conn.commit()
        st.success("已清空資料庫")
    size = conn.execute("PRAGMA page_count").fetchone()[0] * conn.execute("PRAGMA page_size").fetchone()[0]
    st.info(f"資料庫大小：約 {size / 1024:.2f} KB")
    if st.button("下載完整資料表"):
        df_all = pd.read_sql_query("SELECT * FROM records", conn)
        csv = df_all.to_csv(index=False).encode('utf-8')
        st.download_button("下載 CSV", csv, "baccarat_all_records.csv", "text/csv")

st.caption("© 2025 🎲 AI 百家樂 ML 自動化預測下注系統 完整部署版")
