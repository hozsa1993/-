import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ===== 頁面設定 =====
st.set_page_config(page_title="🎲 AI 百家樂 ML 自動化預測下注系統 🎲", page_icon="🎰", layout="centered")

# ===== 激活碼與管理員設定 =====
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
            st.error("密碼錯誤，請重試")
    st.stop()

# ===== 資料庫連線與初始化 =====
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

# ===== Session 狀態初始化 =====
def init_session():
    defaults = {"history": [], "profit": 0, "wins": 0, "total": 0, "base_bet": 100, "current_bet": 100, "auto_bet": False, "max_loss": -1000, "strategy": "固定下注"}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
init_session()

# ===== ML 預測模型準備 =====
df = pd.read_sql_query("SELECT * FROM records ORDER BY created ASC", conn)
df = df[df['result'].isin(['B', 'P', 'T'])].copy()
df['result_code'] = df['result'].map({'B': 1, 'P': 0, 'T': 2})

N = 5
features, labels = [], []
for i in range(len(df) - N):
    features.append(df['result_code'].iloc[i:i + N])
    labels.append(df['result_code'].iloc[i + N])

X, y = np.array(features), np.array(labels)
model, accuracy, can_predict = None, None, False

if len(X) >= 15:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    can_predict = True
else:
    st.warning("歷史資料不足，請先輸入至少 15 筆資料以啟用 ML 預測")

# ===== 預測函數 =====
def ml_predict():
    if not can_predict or len(st.session_state.history) < N:
        return "觀望", 0.0
    mapping = {'B': 1, 'P': 0, 'T': 2}
    recent = [mapping.get(x, 0) for x in st.session_state.history[-N:]]
    pred_code = model.predict([recent])[0]
    prob = max(model.predict_proba([recent])[0])
    pred_label = "莊" if pred_code == 1 else "閒" if pred_code == 0 else "和"
    return pred_label, prob

# ===== 策略調整 =====
st.selectbox("選擇下注策略", ["固定下注", "馬丁格爾", "反馬丁格爾"], key="strategy")
bet_input = st.number_input("設定初始下注金額", min_value=10, step=10, value=st.session_state.base_bet)
st.session_state.base_bet = bet_input
if st.session_state.current_bet < 1:
    st.session_state.current_bet = st.session_state.base_bet

# ===== 自動下注 =====
def auto_bet(pred_label, pred_prob, threshold=0.65):
    if pred_prob < threshold:
        return "信心不足，暫不下注"
    if st.session_state.profit <= st.session_state.max_loss:
        st.warning("已達最大虧損限制，停止自動下注")
        st.session_state.auto_bet = False
        return "已停止下注"
    apply_bet_adjustment(True)
    st.session_state.history.append(pred_label[0])
    c.execute("INSERT INTO records (result, predict, confidence, profit, created) VALUES (?, ?, ?, ?, ?)",
              (pred_label[0], pred_label, pred_prob, 0, datetime.datetime.now()))
    conn.commit()
    return f"已自動下注：{pred_label}"

# ===== 策略下注調整 =====
def apply_bet_adjustment(win):
    if st.session_state.strategy == "固定下注":
        st.session_state.current_bet = st.session_state.base_bet
    elif st.session_state.strategy == "馬丁格爾":
        if win:
            st.session_state.current_bet = st.session_state.base_bet
        else:
            st.session_state.current_bet *= 2
    elif st.session_state.strategy == "反馬丁格爾":
        if win:
            st.session_state.current_bet = min(st.session_state.current_bet * 2, 10000)
        else:
            st.session_state.current_bet = st.session_state.base_bet

# ===== 前端顯示 =====
st.title("🎲 AI 百家樂 ML 自動化預測下注系統 🎲")
if can_predict:
    pred_label, pred_prob = ml_predict()
    st.subheader(f"預測結果：{pred_label} (信心 {pred_prob:.2f})")
    st.write(f"模型準確度：{accuracy:.2%}")
else:
    st.info("尚無法進行預測")

st.checkbox("啟用自動下注 (信心閾值 0.65)", key="auto_bet")
st.session_state.max_loss = st.number_input("當日最大虧損限制 (元)", value=st.session_state.max_loss)

if st.session_state.auto_bet and can_predict:
    st.success(auto_bet(pred_label, pred_prob))

col1, col2 = st.columns(2)
with col1:
    if st.button("✅ 勝利"):
        st.session_state.profit += st.session_state.current_bet
        st.session_state.wins += 1
        st.session_state.total += 1
        apply_bet_adjustment(True)
        st.experimental_rerun()

with col2:
    if st.button("❌ 失敗"):
        st.session_state.profit -= st.session_state.current_bet
        st.session_state.total += 1
        apply_bet_adjustment(False)
        st.experimental_rerun()

st.info(f"目前總獲利：{st.session_state.profit} 元 | 勝場：{st.session_state.wins} | 總場次：{st.session_state.total} | 當前下注金額：{st.session_state.current_bet} 元")

# ===== 管理員後台 =====
if st.session_state.is_admin:
    st.header("🛠️ 管理員後台")
    if st.button("清空資料庫"):
        c.execute("DELETE FROM records")
        conn.commit()
        st.success("已清空資料庫")
    db_size = conn.execute("PRAGMA page_count").fetchone()[0] * conn.execute("PRAGMA page_size").fetchone()[0] / 1024
    st.info(f"資料庫大小：約 {db_size:.2f} KB")
    df_all = pd.read_sql_query("SELECT * FROM records", conn)
    st.download_button("下載完整資料 (CSV)", df_all.to_csv(index=False).encode('utf-8'), "baccarat_records.csv", "text/csv")

st.caption("© 2025 🎲 AI 百家樂 ML 自動化預測下注系統 完整版 (含馬丁格爾、反馬丁格爾)")

