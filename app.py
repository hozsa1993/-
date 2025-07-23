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
    password_input = st.text_input("輸入激活碼/管理員密碼", type="password", key="input_password")
    if st.button("確認", key="btn_confirm_password"):
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
if "history" not in st.session_state:
    st.session_state.history = []

# === ML 三分類模型訓練 ===
def train_rf_model():
    df = pd.read_sql_query("SELECT * FROM records ORDER BY created ASC", conn)
    df = df[df['result'].isin(['B', 'P', 'T'])].copy()
    if len(df) < 30:
        return None, 0.0
    code_map = {'T':0, 'P':1, 'B':2}
    df['result_code'] = df['result'].map(code_map)
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

def ml_predict_probs(model, history):
    if model is None or len(history) < 5:
        return "觀望", 0.0, {"莊": 0.0, "閒": 0.0, "和": 0.0}
    code_map = {'T':0, 'P':1, 'B':2}
    recent = [code_map.get(x.strip(), 0) for x in history[-5:]]
    proba = model.predict_proba([recent])[0]
    pred_idx = model.predict([recent])[0]
    label_map = {0:"和", 1:"閒", 2:"莊"}
    st.write(f"輸入特徵：{recent}")
    st.write(f"機率：莊 {proba[2]:.3f}, 閒 {proba[1]:.3f}, 和 {proba[0]:.3f}")
    probs = {
        "莊": proba[2],
        "閒": proba[1],
        "和": proba[0]
    }
    return label_map[pred_idx], max(proba), probs

# === 讀取資料及模型初始化 ===
df_records = pd.read_sql_query("SELECT * FROM records ORDER BY created ASC", conn)
if "model" not in st.session_state:
    model, model_acc = train_rf_model()
    st.session_state.model = model
    st.session_state.model_acc = model_acc
else:
    model = st.session_state.model
    model_acc = st.session_state.model_acc

st.write(f"資料庫筆數: {len(df_records)}")
if model is not None:
    st.write(f"模型準確率: {model_acc:.2%}")
else:
    st.write("模型尚未訓練")

# === 重新訓練模型按鈕 ===
if st.button("重新訓練模型", key="btn_retrain"):
    model, model_acc = train_rf_model()
    st.session_state.model = model
    st.session_state.model_acc = model_acc
    st.success(f"模型重新訓練完成，準確率：{model_acc:.2%}")

# === 歷史結果輸入（按鈕版）===
st.subheader("輸入最近局結果（點按按鈕加入歷史）")
col1, col2, col3, col4 = st.columns([1,1,1,1])
if col1.button("莊 (B)", key="btn_history_b"):
    st.session_state.history.append("B")
if col2.button("閒 (P)", key="btn_history_p"):
    st.session_state.history.append("P")
if col3.button("和 (T)", key="btn_history_t"):
    st.session_state.history.append("T")
if col4.button("清除歷史", key="btn_history_clear"):
    st.session_state.history = []

st.write("目前歷史結果：", ", ".join(st.session_state.history))
history = st.session_state.history

# === 預測與顯示機率 ===
if len(history) < 5:
    st.warning("請至少輸入 5 局有效結果以供模型預測")
    pred_label, pred_conf, probs = "觀望", 0.0, {"莊":0, "閒":0, "和":0}
else:
    pred_label, pred_conf, probs = ml_predict_probs(model, history)

st.title(f"🎲 AI 百家樂 ML 預測系統 🎲 (RF 三分類模型 準確度: {model_acc:.2%})")
st.markdown("### 預測機率")
st.write(f"莊機率：{probs['莊']*100:.2f}%  |  閒機率：{probs['閒']*100:.2f}%  |  和機率：{probs['和']*100:.2f}%")

if st.button("🔮 預測下一局", key="btn_predict"):
    if pred_conf < 0.6:
        st.info(f"信心不足 ({pred_conf:.2f})，建議觀望")
    else:
        st.success(f"預測：{pred_label} (信心 {pred_conf:.2f})")
    send_signal(f"🎲 預測：{pred_label} (信心 {pred_conf:.2f})")

# === 自動下注與盈虧計算 ===
st.subheader("🎯 自動下注與盈虧管理")
bet_amount = st.number_input("每注金額", min_value=10, value=st.session_state.bet_amount, key="num_bet_amount")
strategy = st.selectbox("選擇下注策略", ["固定下注", "馬丁格爾", "反馬丁格爾"], index=0, key="select_strategy")

col1, col2, col3 = st.columns(3)
clicked_b = col1.button("莊 (B)", key="btn_execute_bet_b")
clicked_p = col2.button("閒 (P)", key="btn_execute_bet_p")
clicked_t = col3.button("和 (T)", key="btn_execute_bet_t")

def calculate_profit_real(pred, actual, bet):
    if actual == "T":
        return 0
    if pred == "莊" and actual == "B":
        return bet * 0.95
    elif pred == "閒" and actual == "P":
        return bet
    else:
        return -bet

def update_bet_amount(strategy, last_profit, base_bet):
    if strategy == "固定下注":
        return base_bet
    elif strategy == "馬丁格爾":
        if last_profit > 0:
            return base_bet
        else:
            return min(base_bet * 2, 100000)
    elif strategy == "反馬丁格爾":
        if last_profit > 0:
            return min(base_bet * 2, 100000)
        else:
            return base_bet

if clicked_b or clicked_p or clicked_t:
    actual_result = "B" if clicked_b else ("P" if clicked_p else "T")
    profit = calculate_profit_real(pred_label, actual_result, bet_amount)
    st.session_state.profit += profit
    st.success(f"下注結果: 預測{pred_label}, 實際{actual_result}, 本次盈虧: {profit}, 總盈虧: {st.session_state.profit}")

    st.session_state.bet_amount = update_bet_amount(strategy, profit, bet_amount)

    c.execute(
        "INSERT INTO records (result, predict, confidence, bet_amount, profit, created) VALUES (?,?,?,?,?,?)",
        (actual_result, pred_label, pred_conf, bet_amount, profit, datetime.datetime.now()))
    conn.commit()

    send_signal(f"已下注: 預測{pred_label}, 實際{actual_result}, 金額: {bet_amount}, 盈虧: {profit}, 總盈虧: {st.session_state.profit}")

    # 自動重新訓練模型
    model, model_acc = train_rf_model()
    st.session_state.model = model
    st.session_state.model_acc = model_acc

# === 策略回測 ===
st.subheader("📊 策略回測")
uploaded_file = st.file_uploader("上傳CSV檔進行回測", key="file_uploader")
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

# === 走勢圖 (預測信心度 & 累積盈虧) ===
def plot_trends(df):
    import matplotlib.ticker as ticker

    if df.empty:
        st.info("無歷史資料，無法繪製走勢圖")
        return

    df = df.sort_values('created').reset_index(drop=True)
    df['cumulative_profit'] = df['profit'].cumsum()
    df['predict_conf'] = df['confidence']

    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.set_xlabel("局數")
    ax1.set_ylabel("預測信心度", color='tab:blue')
    ax1.plot(df.index + 1, df['predict_conf'], label="預測信心度", color='tab:blue', marker='o')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0, 1.05)
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

    ax2 = ax1.twinx()
    ax2.set_ylabel("累積盈虧", color='tab:red')
    ax2.plot(df.index + 1, df['cumulative_profit'], label="累積盈虧", color='tab:red', marker='x')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.suptitle("預測信心度與累積盈虧走勢圖")
    fig.tight_layout()
    st.pyplot(fig)

st.subheader("📈 走勢圖 (預測信心度 & 累積盈虧)")
plot_trends(df_records)

# === 管理員後台 ===
if st.session_state.is_admin:
    with st.expander("🛠️ 管理員後台"):
        if st.button("清空資料庫", key="btn_clear_db"):
            c.execute("DELETE FROM records")
            conn.commit()
            st.success("資料庫已清空")
        df_all = pd.read_sql_query("SELECT * FROM records", conn)
        csv = df_all.to_csv(index=False).encode('utf-8')
        st.download_button("下載完整資料 CSV", csv, "baccarat_records.csv", "text/csv")

st.caption("© 2025 🎲 AI 百家樂 ML 預測系統 | 完整整合版")
