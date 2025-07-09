import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import datetime
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ===== 頁面設定 =====
st.set_page_config(page_title="🎲 AI 百家樂 ML 預測系統 🎲", page_icon="🎰", layout="centered")

# ===== 深色主題 CSS =====
st.markdown("""
<style>
body, .main { background-color: #0f0f0f; color: #e0e0e0; }
button[kind="primary"] { background-color: #FF6F61; color: white; border-radius: 8px; }
hr { border-color: #333333; }
@media (max-width: 768px) {.block-container { padding: 1rem; }}
</style>
""", unsafe_allow_html=True)

# ===== 激活碼鎖定 =====
PASSWORD = "aa17888"
if "access_granted" not in st.session_state:
    st.session_state.access_granted = False

if not st.session_state.access_granted:
    st.markdown("<h1 style='text-align:center; color:#FF6F61;'>🔒 請輸入激活碼以使用系統</h1>", unsafe_allow_html=True)
    password_input = st.text_input("🔑 激活碼", type="password")
    if st.button("確認激活"):
        if password_input == PASSWORD:
            st.session_state.access_granted = True
            st.rerun()
        else:
            st.error("激活碼錯誤，請重新輸入")
    st.stop()

# ===== 資料庫連線與初始化 =====
conn = sqlite3.connect("baccarat.db", check_same_thread=False)
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        result TEXT,
        predict TEXT,
        confidence REAL,
        profit INTEGER,
        created TIMESTAMP
    )
''')
conn.commit()

# ===== Session 狀態初始化 =====
for key, value in {
    "history": [],
    "profit": 0,
    "wins": 0,
    "total": 0,
    "base_bet": 100,
    "current_bet": 100
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ===== 讀取歷史資料並準備訓練資料 =====
df = pd.read_sql_query("SELECT * FROM records ORDER BY created ASC", conn)
df = df[df['result'].isin(['B', 'P'])].copy()
df['result_code'] = df['result'].map({'B': 1, 'P': 0})

N = 5
results = df['result_code'].values
features, labels = [], []
for i in range(len(results) - N):
    features.append(results[i:i + N])
    labels.append(results[i + N])

X, y = np.array(features), np.array(labels)
model, accuracy, can_predict = None, None, False

if len(X) >= 10:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    can_predict = True
else:
    st.warning("歷史資料不足，無法訓練 ML 模型，請先輸入至少 15 筆莊閒結果。")

def ml_predict(history):
    if model is None or len(history) < N:
        return "觀望", 0.0
    code_map = {'B': 1, 'P': 0, 'T': 0}
    recent_codes = [code_map.get(x, 0) for x in history[-N:]]
    pred_code = model.predict([recent_codes])[0]
    pred_prob = max(model.predict_proba([recent_codes])[0])
    return ('莊' if pred_code == 1 else '閒'), pred_prob

st.markdown("<h1 style='text-align:center; color:#FF6F61;'>🎲 AI 百家樂 ML 預測系統 🎲</h1>", unsafe_allow_html=True)
st.divider()

if can_predict:
    pred_label, pred_conf = ml_predict(st.session_state.history)
    st.subheader(f"🔮 機器學習預測建議：{pred_label} (信心 {pred_conf:.2f})")
    st.write(f"模型準確度（測試集）: {accuracy:.2%}")
else:
    st.subheader("🔮 預測：資料不足，無法進行 ML 預測")

st.subheader("🎮 輸入本局結果")
col1, col2, col3 = st.columns(3)

def insert_result(result):
    pred_label, pred_conf = ml_predict(st.session_state.history) if can_predict else ("N/A", 0)
    c.execute(
        "INSERT INTO records (result, predict, confidence, profit, created) VALUES (?, ?, ?, ?, ?)",
        (result, pred_label, pred_conf, 0, datetime.datetime.now().isoformat())
    )
    conn.commit()
    st.rerun()

with col1:
    if st.button("🟥 莊 (B)"):
        st.session_state.history.append("B")
        insert_result("B")

with col2:
    if st.button("🟦 閒 (P)"):
        st.session_state.history.append("P")
        insert_result("P")

with col3:
    if st.button("🟩 和 (T)"):
        st.session_state.history.append("T")
        insert_result("T")

strategy = st.radio("選擇下注策略", ("固定下注", "馬丁格爾", "反馬丁格爾"))
bet_input = st.number_input("設定初始下注金額", min_value=1, step=10, value=st.session_state.base_bet)
st.session_state.base_bet = bet_input
if st.session_state.current_bet < 1:
    st.session_state.current_bet = st.session_state.base_bet

st.subheader(f"💰 勝負紀錄 (目前下注金額: {st.session_state.current_bet} 元)")
col4, col5 = st.columns(2)

with col4:
    if st.button("✅ 勝利"):
        st.session_state.profit += st.session_state.current_bet
        st.session_state.wins += 1
        st.session_state.total += 1
        if strategy == "馬丁格爾":
            st.session_state.current_bet = st.session_state.base_bet
        elif strategy == "反馬丁格爾":
            st.session_state.current_bet = min(st.session_state.current_bet * 2, 10000)
        st.rerun()

with col5:
    if st.button("❌ 失敗"):
        st.session_state.profit -= st.session_state.current_bet
        st.session_state.total += 1
        if strategy == "馬丁格爾":
            st.session_state.current_bet *= 2
        elif strategy == "反馬丁格爾":
            st.session_state.current_bet = max(st.session_state.current_bet // 2, 1)
        st.rerun()

st.success(f"總獲利：{st.session_state.profit} 元 ｜ 勝場：{st.session_state.wins} ｜ 總場：{st.session_state.total}")

st.subheader("📈 近 30 局走勢圖")
if st.session_state.history:
    mapping = {"B": 1, "P": 0, "T": 0.5}
    data = [mapping.get(x, 0) for x in st.session_state.history[-30:]]
    fig, ax = plt.subplots()
    ax.plot(data, marker='o', color='#FF6F61', linestyle='-', linewidth=2)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["閒", "和", "莊"])
    ax.set_xlabel("局數")
    ax.set_title("近 30 局走勢圖")
    st.pyplot(fig)
else:
    st.info("尚無資料")

st.subheader("📥 下載當日紀錄 Excel")
df_today = pd.read_sql_query("SELECT * FROM records WHERE date(created) = date('now')", conn)
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
    df_today.to_excel(writer, index=False, sheet_name='Baccarat Records')
buffer.seek(0)
st.download_button(
    label="📥 下載當日 Excel",
    data=buffer.getvalue(),
    file_name=f"baccarat_records_{datetime.date.today().strftime('%Y%m%d')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.caption("© 2025 🎲 AI 百家樂預測系統 | ML 完整部署修復版")
