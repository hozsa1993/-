import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import datetime
import threading
import os
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
st.set_page_config(page_title="🎲 AI 百家樂 ML 預測系統", page_icon="🎰", layout="wide")

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

# === Session 狀態初始化 ===
if "history" not in st.session_state:
    st.session_state.history = []  # 用來存最近輸入的局結果 ['B','P','T']

if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.model_acc = 0.0

if "profit" not in st.session_state:
    st.session_state.profit = 0.0

if "bet_amount" not in st.session_state:
    st.session_state.bet_amount = 100

if "strategy" not in st.session_state:
    st.session_state.strategy = "固定下注"

# === 特徵工程 ===
def extract_features(results, N=5):
    features = []
    labels = []
    for i in range(N, len(results)):
        window = results[i-N:i]
        label = results[i]
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

        feat = list(window) + [count_b, count_p, count_t, prop_b, prop_p, prop_t,
                              max_consec_b, max_consec_p, max_consec_t]
        features.append(feat)
        labels.append(label)
    return np.array(features), np.array(labels)

# === 訓練模型 ===
def train_model():
    df = pd.read_sql_query("SELECT * FROM records WHERE result IN ('B','P','T') ORDER BY created ASC", conn)
    if len(df) < 30:
        return None, 0.0
    code_map = {'T':0, 'P':1, 'B':2}
    df['code'] = df['result'].map(code_map)
    results = df['code'].tolist()

    N = 5
    X, y = extract_features(results, N)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    return model, acc

# === 預測函數 ===
def predict_next(model, history):
    if model is None or len(history) < 5:
        return "觀望", 0.0, {"莊":0.0,"閒":0.0,"和":0.0}
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

    feat = list(recent) + [count_b, count_p, count_t, prop_b, prop_p, prop_t,
                          max_consec_b, max_consec_p, max_consec_t]

    proba = model.predict_proba([feat])[0]
    pred_idx = model.predict([feat])[0]

    label_map = {0:"和", 1:"閒", 2:"莊"}
    probs = {
        "莊": proba[2],
        "閒": proba[1],
        "和": proba[0]
    }
    return label_map[pred_idx], max(proba), probs

# === 主頁面 ===
st.title("🎲 AI 百家樂 ML 預測系統")

# 輸入歷史結果按鈕
st.subheader("輸入最近局結果")
col1, col2, col3 = st.columns(3)
if col1.button("莊 (B)"):
    st.session_state.history.append("B")
if col2.button("閒 (P)"):
    st.session_state.history.append("P")
if col3.button("和 (T)"):
    st.session_state.history.append("T")

# 顯示歷史結果
st.write("最近輸入結果（最多50筆）:", ", ".join(st.session_state.history[-50:]))

# 模型訓練與預測
if st.session_state.model is None:
    st.info("模型尚未訓練，請先按下方「重新訓練模型」按鈕")
else:
    pred_label, pred_conf, pred_probs = predict_next(st.session_state.model, st.session_state.history)
    st.subheader("🔮 預測下一局")
    st.write(f"預測結果：**{pred_label}**，信心度：{pred_conf:.2%}")
    st.write(f"莊: {pred_probs['莊']:.2%} | 閒: {pred_probs['閒']:.2%} | 和: {pred_probs['和']:.2%}")

if st.button("重新訓練模型"):
    with st.spinner("模型訓練中..."):
        model, acc = train_model()
        if model:
            st.session_state.model = model
            st.session_state.model_acc = acc
            st.success(f"模型訓練完成，準確度：{acc:.2%}")
        else:
            st.warning("資料不足，至少需要30筆資料才能訓練模型")

# 下注金額及策略
st.subheader("下注設定")
bet_amount = st.number_input("每注金額", min_value=10, value=st.session_state.bet_amount)
strategy = st.selectbox("下注策略", ["固定下注", "馬丁格爾", "反馬丁格爾"], index=0)

def calculate_profit(pred_label, bet_amount):
    if pred_label == "莊":
        return bet_amount * 0.95
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

if st.button("執行下注"):
    if st.session_state.model is None:
        st.error("模型未訓練，無法下注")
    else:
        pred_label, pred_conf, _ = predict_next(st.session_state.model, st.session_state.history)
        profit = calculate_profit(pred_label, bet_amount)
        st.session_state.profit += profit
        st.session_state.bet_amount = update_bet_amount(strategy, profit)
        st.success(f"下注結果：{pred_label}，本次盈虧：{profit}，累積盈虧：{st.session_state.profit}")

        # 寫入資料庫，result 預設用「待填」
        c.execute("INSERT INTO records (result, predict, confidence, bet_amount, profit, created) VALUES (?,?,?,?,?,?)",
                  ("待填", pred_label, pred_conf, bet_amount, profit, datetime.datetime.now()))
        conn.commit()

# 歷史結果走勢圖
if len(st.session_state.history) > 0:
    st.subheader("歷史結果走勢圖")
    code_map = {'B': 2, 'P': 1, 'T': 0}
    history_nums = [code_map.get(x, 0) for x in st.session_state.history]
    plt.figure(figsize=(12,3))
    plt.plot(history_nums, marker='o')
    plt.yticks([0,1,2], ['和(T)', '閒(P)', '莊(B)'])
    plt.title("歷史結果走勢")
    plt.grid(True)
    st.pyplot(plt)

# 管理員功能
if st.session_state.is_admin:
    with st.expander("管理員後台"):
        if st.button("清空資料庫"):
            c.execute("DELETE FROM records")
            conn.commit()
            st.success("資料庫已清空")
        df_all = pd.read_sql_query("SELECT * FROM records", conn)
        csv = df_all.to_csv(index=False).encode('utf-8')
        st.download_button("下載完整資料 CSV", csv, "baccarat_records.csv", "text/csv")

st.caption("© 2025 AI 百家樂 ML 預測系統")

