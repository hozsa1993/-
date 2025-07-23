import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import datetime
import time
import threading
import os
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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

# === 資料庫初始化 ===
conn = sqlite3.connect("baccarat.db", check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    result TEXT,
    predict TEXT,
    confidence REAL,
    bet_amount REAL,
    profit REAL,
    created TIMESTAMP
)''')
conn.commit()

# === Telegram 推播設定 ===
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"

def send_signal(message):
    if TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN" or TELEGRAM_CHAT_ID == "YOUR_TELEGRAM_CHAT_ID":
        print(f"[模擬推播] {message}")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=data)
        if response.status_code != 200:
            print(f"[推播錯誤] {response.text}")
    except Exception as e:
        print(f"[推播例外] {e}")

# === Session 狀態初始值 ===
if "profit" not in st.session_state:
    st.session_state.profit = 0
if "bet_amount" not in st.session_state:
    st.session_state.bet_amount = 100
if "strategy" not in st.session_state:
    st.session_state.strategy = "固定下注"
if "current_bet" not in st.session_state:
    st.session_state.current_bet = st.session_state.bet_amount

# === ML 多模型簡化示範：只用 RF 模型 ===
def train_rf_model():
    df = pd.read_sql_query("SELECT * FROM records ORDER BY created ASC", conn)
    df = df[df['result'].isin(['B', 'P'])].copy()
    if len(df) < 15:
        return None, 0.0
    df['result_code'] = df['result'].map({'B': 1, 'P': 0})
    N = 5
    features, labels = [], []
    results = df['result_code'].values
    for i in range(len(results) - N):
        features.append(results[i:i + N])
        labels.append(results[i + N])
    X, y = np.array(features), np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return model, accuracy

def ml_predict(model, history):
    if model is None or len(history) < 5:
        return "觀望", 0.0
    code_map = {'B': 1, 'P': 0, 'T': 0}
    recent = [code_map.get(x.strip(), 0) for x in history[-5:]]
    pred = model.predict([recent])[0]
    prob = max(model.predict_proba([recent])[0])
    return ("莊" if pred == 1 else "閒"), prob

model, model_acc = train_rf_model()

# === 頁面標題與預測 ===
st.title(f"🎲 AI 百家樂 ML 預測系統 🎲 (RF 模型 準確度: {model_acc:.2%})")

history = st.text_area("輸入最近局結果 (B,P,T 以逗號分隔)").strip().split(",")
pred_label, pred_conf = ml_predict(model, history)

if st.button("🔮 預測下一局"):
    if pred_conf < 0.6:
        st.info(f"信心不足 ({pred_conf:.2f})，建議觀望")
    else:
        st.success(f"預測：{pred_label} (信心 {pred_conf:.2f})")
    send_signal(f"🎲 預測：{pred_label} (信心 {pred_conf:.2f})")

# === 自動下注與盈虧計算 ===
st.subheader("🎯 自動下注與盈虧管理")
bet_amount = st.number_input("每注金額", min_value=10, value=st.session_state.bet_amount)
strategy = st.selectbox("選擇下注策略", ["固定下注", "馬丁格爾", "反馬丁格爾"], index=0)

def calculate_profit(pred_label, bet_amount):
    if pred_label == "莊":
        return bet_amount * 0.95  # 抽水5%
    elif pred_label == "閒":
        return bet_amount
    else:
        return 0

def update_bet_amount(strategy, last_profit):
    if strategy == "固定下注":
        return bet_amount
    elif strategy == "馬丁格爾":
        if last_profit > 0:
            return bet_amount
        else:
            return min(bet_amount * 2, 100000)
    elif strategy == "反馬丁格爾":
        if last_profit > 0:
            return min(bet_amount * 2, 100000)
        else:
            return bet_amount

if st.button("✅ 執行下注"):
    profit = calculate_profit(pred_label, bet_amount)
    st.session_state.profit += profit
    st.success(f"下注結果: {pred_label}, 本次盈虧: {profit}, 總盈虧: {st.session_state.profit}")
    st.session_state.bet_amount = update_bet_amount(strategy, profit)
    c.execute("INSERT INTO records (result, predict, confidence, bet_amount, profit, created) VALUES (?,?,?,?,?,?)",
              ("待填", pred_label, pred_conf, bet_amount, profit, datetime.datetime.now()))
    conn.commit()
    send_signal(f"已下注: {pred_label}, 金額: {bet_amount}, 盈虧: {profit}, 總盈虧: {st.session_state.profit}")

# === 策略回測 ===
st.subheader("📊 策略回測")
uploaded_file = st.file_uploader("上傳CSV檔進行回測")
def backtest_strategy(df, strategy):
    df = df.copy()
    df['cumulative_profit'] = 0
    profit = 0
    bet = 100
    profits = []
    for idx, row in df.iterrows():
        if strategy == "固定下注":
            bet = 100
        elif strategy == "馬丁格爾":
            bet = 100 if profit > 0 else min(bet * 2, 100000)
        elif strategy == "反馬丁格爾":
            bet = min(bet * 2, 100000) if profit > 0 else 100

        profit += row['profit']
        profits.append(profit)
        df.at[idx, 'cumulative_profit'] = profit

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(profits, label='累積盈虧')
    ax.set_xlabel('局數')
    ax.set_ylabel('累積盈虧')
    ax.set_title(f'{strategy} 策略回測盈虧曲線')
    ax.grid(True)
    ax.legend()
    return fig

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    fig = backtest_strategy(df, strategy)
    st.pyplot(fig)

# === 管理員後台 ===
if st.session_state.is_admin:
    with st.expander("🛠️ 管理員後台"):
        if st.button("清空資料庫"):
            c.execute("DELETE FROM records")
            conn.commit()
            st.success("資料庫已清空")
        df_all = pd.read_sql_query("SELECT * FROM records", conn)
        csv = df_all.to_csv(index=False).encode('utf-8')
        st.download_button("下載完整資料 CSV", csv, "baccarat_records.csv", "text/csv")

st.caption("© 2025 🎲 AI 百家樂 ML 預測系統 | 完整整合版")

