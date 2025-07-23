import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import datetime
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
st.set_page_config(page_title="🎲 AI 百家樂 ML 預測系統 強化版 🎲", page_icon="🎰", layout="wide")

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

# === Telegram 推播設定（選填）===
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

if "history" not in st.session_state:
    st.session_state.history = []  # 用於存儲歷史輸入結果 ['B','P','T']

# === 特徵工程函數 ===
def extract_features(results, N=5):
    features = []
    labels = []
    for i in range(N, len(results)):
        window = results[i-N:i]
        label = results[i]
        # 基本序列特徵：過去N局結果
        base = list(window)
        # 新增統計特徵：莊、閒、和 出現次數與比例
        count_b = window.count(2)
        count_p = window.count(1)
        count_t = window.count(0)
        prop_b = count_b / N
        prop_p = count_p / N
        prop_t = count_t / N

        def max_consecutive(seq, val):
            max_len = cur_len = 0
            for x in seq:
                if x == val:
                    cur_len += 1
                    max_len = max(max_len, cur_len)
                else:
                    cur_len = 0
            return max_len

        max_consec_b = max_consecutive(window, 2)
        max_consec_p = max_consecutive(window, 1)
        max_consec_t = max_consecutive(window, 0)

        feat = base + [count_b, count_p, count_t, prop_b, prop_p, prop_t,
                      max_consec_b, max_consec_p, max_consec_t]
        features.append(feat)
        labels.append(label)
    return np.array(features), np.array(labels)

# === 訓練模型 ===
def train_rf_model_enhanced():
    df = pd.read_sql_query("SELECT * FROM records ORDER BY created ASC", conn)
    df = df[df['result'].isin(['B', 'P', 'T'])].copy()
    if len(df) < 30:
        return None, 0.0
    code_map = {'T':0, 'P':1, 'B':2}
    df['result_code'] = df['result'].map(code_map)
    results = df['result_code'].tolist()

    N = 5
    X, y = extract_features(results, N)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return model, accuracy

# === 預測函數 ===
def ml_predict_probs_enhanced(model, history):
    if model is None or len(history) < 5:
        return "觀望", 0.0, {"莊": 0.0, "閒": 0.0, "和": 0.0}
    code_map = {'T':0, 'P':1, 'B':2}
    recent = [code_map.get(x.strip(), 0) for x in history[-5:]]

    count_b = recent.count(2)
    count_p = recent.count(1)
    count_t = recent.count(0)
    prop_b = count_b / 5
    prop_p = count_p / 5
    prop_t = count_t / 5

    def max_consecutive(seq, val):
        max_len = cur_len = 0
        for x in seq:
            if x == val:
                cur_len += 1
                max_len = max(max_len, cur_len)
            else:
                cur_len = 0
        return max_len

    max_consec_b = max_consecutive(recent, 2)
    max_consec_p = max_consecutive(recent, 1)
    max_consec_t = max_consecutive(recent, 0)

    feat = recent + [count_b, count_p, count_t, prop_b, prop_p, prop_t,
                    max_consec_b, max_consec_p, max_consec_t]

    proba = model.predict_proba([feat])[0]
    pred_idx = model.predict([feat])[0]
    label_map = {0: "和", 1: "閒", 2: "莊"}
    probs = {
        "莊": proba[2],
        "閒": proba[1],
        "和": proba[0]
    }
    return label_map[pred_idx], max(proba), probs

# === 載入模型 ===
model, model_acc = train_rf_model_enhanced()

# === 頁面標題 ===
st.title(f"🎲 AI 百家樂 ML 預測系統 強化版 🎲 (準確度: {model_acc:.2%})")

# === 歷史走勢與輸入按鈕 ===
st.subheader("📈 歷史走勢輸入")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("莊 (B)"):
        st.session_state.history.append('B')
with col2:
    if st.button("閒 (P)"):
        st.session_state.history.append('P')
with col3:
    if st.button("和 (T)"):
        st.session_state.history.append('T')

# 顯示目前歷史記錄（最新50筆）
history_display = st.session_state.history[-50:]
st.write("最近結果:", ", ".join(history_display))

# === 預測結果 ===
pred_label, pred_conf, pred_probs = ml_predict_probs_enhanced(model, st.session_state.history)

st.subheader("🔮 預測下一局結果")
st.write(f"預測結果：**{pred_label}**，信心度：{pred_conf:.2%}")

st.write("各類機率：")
st.write(f"莊: {pred_probs['莊']:.2%} | 閒: {pred_probs['閒']:.2%} | 和: {pred_probs['和']:.2%}")

# === 重新訓練模型按鈕 ===
if st.button("🔄 重新訓練模型"):
    with st.spinner("訓練中，請稍候..."):
        model, model_acc = train_rf_model_enhanced()
        st.success(f"模型重新訓練完成！準確度：{model_acc:.2%}")

# === 自動下注與盈虧管理 ===
st.subheader("🎯 自動下注與盈虧管理")
bet_amount = st.number_input("每注金額", min_value=10, value=st.session_state.bet_amount)
strategy = st.selectbox("選擇下注策略", ["固定下注", "馬丁格爾", "反馬丁格爾"], index=0)

def calculate_profit(pred_label, bet_amount):
    # 模擬結果，真實要連動資料庫或實際結果
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
    # 寫入資料庫，這裡暫用 "待填" 代表真實結果
    c.execute("INSERT INTO records (result, predict, confidence, bet_amount, profit, created) VALUES (?,?,?,?,?,?)",
              ("待填", pred_label, pred_conf, bet_amount, profit, datetime.datetime.now()))
    conn.commit()
    send_signal(f"已下注: {pred_label}, 金額: {bet_amount}, 盈虧: {profit}, 總盈虧: {st.session_state.profit}")

# === 走勢圖顯示 ===
st.subheader("📊 歷史結果走勢圖")

if len(st.session_state.history) > 0:
    code_map = {'B': 2, 'P': 1, 'T': 0}
    history_nums = [code_map.get(x, 0) for x in st.session_state.history]
    plt.figure(figsize=(12, 3))
    plt.plot(history_nums, marker='o')
    plt.yticks([0,1,2], ['和(T)', '閒(P)', '莊(B)'])
    plt.title("歷史結果走勢")
    plt.grid(True)
    st.pyplot(plt)

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

st.caption("© 2025 🎲 AI 百家樂 ML 預測系統 強化版 | 完整整合版")

