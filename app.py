import os
import time
import datetime
import threading
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import io
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# === 自動每日凌晨4點重啟防卡死 ===
def daily_reload(hour=4):
    while True:
        now = datetime.datetime.now()
        target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        if now > target:
            target += datetime.timedelta(days=1)
        wait_sec = (target - now).total_seconds()
        print(f"[AutoReload] 等待 {int(wait_sec)} 秒，將於 {target} 重啟程式")
        time.sleep(wait_sec)
        os._exit(0)

threading.Thread(target=daily_reload, daemon=True).start()

# === 頁面設定 ===
st.set_page_config(page_title="🎲 AI 百家樂 ML 預測系統 🎲", page_icon="🎰", layout="wide")

# === 激活碼設定 ===
ADMIN_PASSWORD = "admin999"
PASSWORDS = {"aa17888", "bb12345"}

if "access_granted" not in st.session_state:
    st.session_state.access_granted = False
    st.session_state.is_admin = False

if not st.session_state.access_granted:
    st.title("🔒 請輸入激活碼或管理員密碼")
    password_input = st.text_input("輸入激活碼/管理員密碼", type="password")
    if st.button("確認"):
        if password_input in PASSWORDS:
            st.session_state.access_granted = True
        elif password_input == ADMIN_PASSWORD:
            st.session_state.access_granted = True
            st.session_state.is_admin = True
        else:
            st.error("密碼錯誤，請重試")
    st.stop()

# === 資料庫初始化 ===
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

# === Session 狀態 ===
def init_session():
    default = {
        "history": [],
        "profit": 0,
        "wins": 0,
        "total": 0,
        "base_bet": 100,
        "current_bet": 100,
        "auto_bet": False,
        "max_loss": -1000,
        "strategy": "固定下注",
        "confidence_threshold": 0.65
    }
    for k, v in default.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# === ML 模型 ===
df = pd.read_sql_query("SELECT * FROM records ORDER BY created ASC", conn)
df = df[df['result'].isin(['B', 'P'])].copy()
df['result_code'] = df['result'].map({'B': 1, 'P': 0})
N = 5
features, labels = [], []
results = df['result_code'].values
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

def ml_predict(history):
    if model is None or len(history) < N:
        return "觀望", 0.0
    code_map = {'B': 1, 'P': 0, 'T': 0}
    recent = [code_map.get(x, 0) for x in history[-N:]]
    pred = model.predict([recent])[0]
    prob = max(model.predict_proba([recent])[0])
    return ("莊" if pred == 1 else "閒"), prob

# === 顯示標題 ===
st.title("🎲 AI 百家樂 ML 預測系統 🎲")

# === 預測顯示 ===
if can_predict:
    pred_label, pred_conf = ml_predict(st.session_state.history)
    if pred_conf < st.session_state.confidence_threshold:
        st.info(f"🔮 信心不足 ({pred_conf:.2f})，建議觀望")
    else:
        st.success(f"🔮 預測建議：{pred_label} (信心 {pred_conf:.2f})")
    st.caption(f"模型準確度：{accuracy:.2%}")
else:
    st.warning("資料不足，需至少 15 筆資料以啟用預測")

# === 輸入結果 ===
col1, col2, col3 = st.columns(3)
def insert_result(result):
    pred_label, pred_conf = ml_predict(st.session_state.history) if can_predict else ("N/A", 0.0)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO records (result, predict, confidence, profit, created) VALUES (?, ?, ?, ?, ?)",
              (result, pred_label, pred_conf, 0, now))
    conn.commit()
    st.session_state.history.append(result)
    st.rerun()

with col1:
    if st.button("🟥 莊 (B)"):
        insert_result("B")
with col2:
    if st.button("🟦 閒 (P)"):
        insert_result("P")
with col3:
    if st.button("🟩 和 (T)"):
        insert_result("T")

# === 策略設定 ===
st.subheader("🎯 下注策略與設定")
st.session_state.strategy = st.radio("選擇下注策略", ["固定下注", "馬丁格爾", "反馬丁格爾"], index=["固定下注", "馬丁格爾", "反馬丁格爾"].index(st.session_state.strategy))
st.session_state.base_bet = st.number_input("初始下注金額", min_value=1, value=st.session_state.base_bet)
st.session_state.max_loss = st.number_input("最大虧損限制", min_value=-1000000, value=st.session_state.max_loss)
st.session_state.confidence_threshold = st.slider("自動下注信心閾值", 0.5, 0.95, st.session_state.confidence_threshold, 0.05)
st.session_state.auto_bet = st.checkbox("啟用自動下注", value=st.session_state.auto_bet)

# === 自動下注執行 ===
def apply_bet(win):
    if st.session_state.strategy == "固定下注":
        st.session_state.current_bet = st.session_state.base_bet
    elif st.session_state.strategy == "馬丁格爾":
        st.session_state.current_bet = st.session_state.base_bet if win else min(st.session_state.current_bet * 2, 500000)
    elif st.session_state.strategy == "反馬丁格爾":
        st.session_state.current_bet = min(st.session_state.current_bet * 2, 500000) if win else st.session_state.base_bet

def auto_bet(pred_label, pred_conf):
    if pred_conf < st.session_state.confidence_threshold:
        return "信心不足，暫不下注"
    if st.session_state.profit <= st.session_state.max_loss:
        st.warning("已達最大虧損限制，停止自動下注")
        st.session_state.auto_bet = False
        return "已停止自動下注"
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO records (result, predict, confidence, profit, created) VALUES (?, ?, ?, ?, ?)",
              (pred_label[0], pred_label, pred_conf, 0, now))
    conn.commit()
    st.session_state.history.append(pred_label[0])
    return f"已自動下注：{pred_label}"

if st.session_state.auto_bet and can_predict:
    st.success(auto_bet(pred_label, pred_conf))

# === 勝敗記錄 ===
st.subheader(f"💰 勝負紀錄 (目前下注：{st.session_state.current_bet} 元)")
col_win, col_lose = st.columns(2)
with col_win:
    if st.button("✅ 勝利"):
        st.session_state.profit += st.session_state.current_bet
        st.session_state.wins += 1
        st.session_state.total += 1
        apply_bet(True)
        st.rerun()
with col_lose:
    if st.button("❌ 失敗"):
        st.session_state.profit -= st.session_state.current_bet
        st.session_state.total += 1
        apply_bet(False)
        st.rerun()
st.success(f"總獲利：{st.session_state.profit} 元 ｜ 勝場：{st.session_state.wins} ｜ 總場次：{st.session_state.total}")

# === 走勢圖 ===
st.subheader("📈 近 30 局走勢圖")
if st.session_state.history:
    mapping = {"B": 1, "P": 0, "T": 0.5}
    data = [mapping.get(x, 0) for x in st.session_state.history[-30:]]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(data, marker='o', color='#FF6F61', linewidth=2)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["閒", "和", "莊"])
    ax.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig)
else:
    st.info("尚無資料可繪製走勢圖")

# === 管理員後台 ===
if st.session_state.is_admin:
    with st.expander("🛠️ 管理員後台"):
        if st.button("清空資料庫"):
            c.execute("DELETE FROM records")
            conn.commit()
            st.success("資料庫已清空")
            st.rerun()
        df_all = pd.read_sql_query("SELECT * FROM records", conn)
        csv = df_all.to_csv(index=False).encode('utf-8')
        st.download_button("下載完整資料 CSV", csv, "baccarat_records.csv", "text/csv")

st.caption("© 2025 🎲 AI 百家樂 ML 預測系統 | 完整版含走勢圖、策略、每日自動重啟、預測、自動下注、管理員後台")
