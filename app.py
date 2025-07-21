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

# ===== 多組激活碼與管理員設定 =====
ADMIN_PASSWORD = "admin999"
PASSWORDS = {"aa17888", "bb12345"}

if "access_granted" not in st.session_state:
    st.session_state.access_granted = False
    st.session_state.is_admin = False

if not st.session_state.access_granted:
    st.markdown("<h1 style='text-align:center; color:#FF6F61;'>🔒 請輸入激活碼或管理員密碼以使用系統</h1>", unsafe_allow_html=True)
    password_input = st.text_input("🔑 激活碼/管理員密碼", type="password", key="password_input")
    if st.button("確認激活"):
        if password_input in PASSWORDS:
            st.session_state.access_granted = True
        elif password_input == ADMIN_PASSWORD:
            st.session_state.access_granted = True
            st.session_state.is_admin = True
        else:
            st.error("激活碼或密碼錯誤，請重新輸入")
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
    "current_bet": 100,
    "auto_bet": False,
    "max_loss": -1000,
    "strategy": "固定下注"
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

# 顯示預測
if can_predict:
    pred_label, pred_conf = ml_predict(st.session_state.history)
    st.subheader(f"🔮 機器學習預測建議：{pred_label} (信心 {pred_conf:.2f})")
    st.write(f"模型準確度（測試集）: {accuracy:.2%}")
else:
    st.subheader("🔮 預測：資料不足，無法進行 ML 預測")

# 輸入本局結果
st.subheader("🎮 輸入本局結果")
col1, col2, col3 = st.columns(3)

def insert_result(result):
    pred_label, pred_conf = ml_predict(st.session_state.history) if can_predict else ("N/A", 0)
    c.execute(
        "INSERT INTO records (result, predict, confidence, profit, created) VALUES (?, ?, ?, ?, ?)",
        (result, pred_label, pred_conf, 0, datetime.datetime.now().isoformat())
    )
    conn.commit()
    st.session_state.history.append(result)
    st.experimental_rerun()

with col1:
    if st.button("🟥 莊 (B)"):
        insert_result("B")

with col2:
    if st.button("🟦 閒 (P)"):
        insert_result("P")

with col3:
    if st.button("🟩 和 (T)"):
        insert_result("T")

# 下注策略設定
strategy = st.radio("選擇下注策略", ["固定下注", "馬丁格爾", "反馬丁格爾"], index=["固定下注", "馬丁格爾", "反馬丁格爾"].index(st.session_state.strategy))
st.session_state.strategy = strategy

# 初始下注金額
bet_input = st.number_input("設定初始下注金額", min_value=1, step=10, value=st.session_state.base_bet)
st.session_state.base_bet = bet_input
if st.session_state.current_bet < 1:
    st.session_state.current_bet = st.session_state.base_bet

# 最大虧損限制
max_loss = st.number_input("當日最大虧損限制(負數)", value=st.session_state.get("max_loss", -1000))
st.session_state.max_loss = max_loss

# 自動下注開關
auto_bet_flag = st.checkbox("啟用自動下注(信心閾值0.65)", value=st.session_state.auto_bet)
st.session_state.auto_bet = auto_bet_flag

# 下注策略調整下注金額函數
def apply_bet_adjustment(win):
    strat = st.session_state.strategy
    if strat == "固定下注":
        st.session_state.current_bet = st.session_state.base_bet
    elif strat == "馬丁格爾":
        if win:
            st.session_state.current_bet = st.session_state.base_bet
        else:
            st.session_state.current_bet = min(st.session_state.current_bet * 2, 100000)
    elif strat == "反馬丁格爾":
        if win:
            st.session_state.current_bet = min(st.session_state.current_bet * 2, 100000)
        else:
            st.session_state.current_bet = st.session_state.base_bet

def auto_bet(pred_label, pred_prob, threshold=0.65):
    if pred_prob < threshold:
        return "信心不足，暫不下注"
    if st.session_state.profit <= st.session_state.max_loss:
        st.warning("已達最大虧損限制，停止自動下注")
        st.session_state.auto_bet = False
        return "已停止下注"
    # 這裡預設贏，但實際結果要用戶輸入後調整下注金額
    st.session_state.history.append(pred_label[0])
    c.execute("INSERT INTO records (result, predict, confidence, profit, created) VALUES (?, ?, ?, ?, ?)",
              (pred_label[0], pred_label, pred_prob, 0, datetime.datetime.now()))
    conn.commit()
    return f"已自動下注：{pred_label}"

if auto_bet_flag and can_predict:
    pred_label, pred_prob = ml_predict(st.session_state.history)
    result_msg = auto_bet(pred_label, pred_prob)
    st.success(result_msg)

# 勝負記錄與下注金額調整
st.subheader(f"💰 勝負紀錄 (目前下注金額: {st.session_state.current_bet} 元)")
col4, col5 = st.columns(2)

with col4:
    if st.button("✅ 勝利"):
        st.session_state.profit += st.session_state.current_bet
        st.session_state.wins += 1
        st.session_state.total += 1
        apply_bet_adjustment(True)
        st.experimental_rerun()

with col5:
    if st.button("❌ 失敗"):
        st.session_state.profit -= st.session_state.current_bet
        st.session_state.total += 1
        apply_bet_adjustment(False)
        st.experimental_rerun()

st.success(f"總獲利：{st.session_state.profit} 元 ｜ 勝場：{st.session_state.wins} ｜ 總場：{st.session_state.total}")

# 近30局走勢圖
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

# 下載當日紀錄 Excel
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

# 管理員後台功能
if st.session_state.is_admin:
    st.header("🛠️ 管理員後台")
    if st.button("清空資料庫"):
        c.execute("DELETE FROM records")
        conn.commit()
        st.success("已清空資料庫")
        st.experimental_rerun()

    db_size_kb = conn.execute("PRAGMA page_count").fetchone()[0] * conn.execute("PRAGMA page_size").fetchone()[0] / 1024
    st.info(f"資料庫大小：約 {db_size_kb:.2f} KB")

    df_all = pd.read_sql_query("SELECT * FROM records", conn)
    st.download_button("下載完整資料 (CSV)", df_all.to_csv(index=False).encode('utf-8'), "baccarat_records.csv", "text/csv")

st.caption("© 2025 🎲 AI 百家樂 ML 預測系統 | 完整整合版 | 含多激活碼、管理員、下注策略與自動下注功能")



