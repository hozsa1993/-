import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import datetime
import io

# ===== 頁面設定 =====
st.set_page_config(page_title="🎲 AI 百家樂預測系統 🎲", page_icon="🎰", layout="centered")

# ===== 深色美化 + 手機適配 =====
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
            st.experimental_rerun()
        else:
            st.error("激活碼錯誤，請重新輸入")
    st.stop()

# ===== 資料庫初始化 =====
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

# ===== Session 初始化 =====
if "history" not in st.session_state:
    st.session_state.history = []
if "profit" not in st.session_state:
    st.session_state.profit = 0
if "wins" not in st.session_state:
    st.session_state.wins = 0
if "total" not in st.session_state:
    st.session_state.total = 0

# ===== 預測邏輯 =====
def predict_next(history):
    from collections import Counter
    if len(history) < 5:
        return "觀望", 0.0
    cnt = Counter(history[-10:])
    if cnt['B'] > cnt['P']:
        return "B", cnt['B'] / 10
    else:
        return "P", cnt['P'] / 10

# ===== 頁面標題 =====
st.markdown("<h1 style='text-align:center; color:#FF6F61;'>🎲 AI 百家樂預測系統 🎲</h1>", unsafe_allow_html=True)
st.divider()

# ===== 預測區 =====
predict, confidence = predict_next(st.session_state.history)
st.subheader(f"🔮 預測建議：{'莊' if predict=='B' else '閒' if predict=='P' else predict} (信心 {confidence:.2f})")

# ===== 輸入結果 =====
st.subheader("🎮 輸入本局結果")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🟥 莊 (B)"):
        st.session_state.history.append("B")
        c.execute("INSERT INTO records (result, predict, confidence, profit, created) VALUES (?, ?, ?, ?, ?)",
                  ("B", predict, confidence, 0, datetime.datetime.now()))
        conn.commit()
with col2:
    if st.button("🟦 閒 (P)"):
        st.session_state.history.append("P")
        c.execute("INSERT INTO records (result, predict, confidence, profit, created) VALUES (?, ?, ?, ?, ?)",
                  ("P", predict, confidence, 0, datetime.datetime.now()))
        conn.commit()
with col3:
    if st.button("🟩 和 (T)"):
        st.session_state.history.append("T")
        c.execute("INSERT INTO records (result, predict, confidence, profit, created) VALUES (?, ?, ?, ?, ?)",
                  ("T", predict, confidence, 0, datetime.datetime.now()))
        conn.commit()

# ===== 勝負紀錄 =====
st.subheader("💰 勝負紀錄")
col4, col5 = st.columns(2)
with col4:
    if st.button("✅ 勝利 +100"):
        st.session_state.profit += 100
        st.session_state.wins += 1
        st.session_state.total += 1
with col5:
    if st.button("❌ 失敗 -100"):
        st.session_state.profit -= 100
        st.session_state.total += 1

st.success(f"總獲利：{st.session_state.profit} 元 ｜ 勝場：{st.session_state.wins} ｜ 總場：{st.session_state.total}")

# ===== 走勢圖 =====
st.subheader("📈 近 30 局走勢圖")
if st.session_state.history:
    mapping = {"B": 1, "P": 0, "T": 0.5}
    data = [mapping[x] for x in st.session_state.history[-30:]]
    fig, ax = plt.subplots()
    ax.plot(data, marker='o', color='#FF6F61', linestyle='-', linewidth=2)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["閒", "和", "莊"])
    ax.set_xlabel("局數")
    ax.set_title("近 30 局走勢圖")
    st.pyplot(fig)
else:
    st.info("尚無資料")

# ===== 下載 Excel 報表 =====
st.subheader("📥 下載當日紀錄 Excel")
df = pd.read_sql_query("SELECT * FROM records WHERE date(created)=date('now')", conn)
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
    df.to_excel(writer, index=False, sheet_name='Baccarat Records')
    writer.save()
st.download_button("📥 下載當日 Excel", data=buffer.getvalue(), file_name="baccarat_records.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("© 2025 🎲 AI 百家樂預測系統 | 加入激活碼保護")
