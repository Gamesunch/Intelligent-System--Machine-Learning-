import streamlit as st

st.title("📊 **การพัฒนาโมเดล LSTM สำหรับพยากรณ์ราคาหุ้น Nvidia**")

st.header("📌 **ที่มาของ Dataset**")
st.markdown("[📂 Nvidia Stock Price Dataset](https://www.kaggle.com/datasets/meharshanali/nvidia-stocks-data-2025)")

st.header("🔹 **มี Feature 7 อันดังนี้**")

features = {
    "📅 Date": "วันที่ของข้อมูล",
    "📈 Adj Close": "ราคาปิดที่ปรับปรุงแล้ว (Adjusted Close Price)",
    "💰 Close": "ราคาปิดของหุ้น",
    "📊 High": "ราคาสูงสุดของวัน",
    "📉 Low": "ราคาต่ำสุดของวัน",
    "🔄 Open": "ราคาเปิดของวัน",
    "📦 Volume": "ปริมาณการซื้อขาย (จำนวนหุ้นที่ซื้อขายในวันนั้น)"
}

for feature, description in features.items():
    st.markdown(f"- **{feature}**: {description}")

st.header("📌 **Data Analysis and Cleaning**")
st.write("ขั้นแรกเราจะมาดู head ของ Dataset กันก่อน")
st.write("Head ของ Dataset")
st.image("image/NN/head.png")
st.write("จากนั้นเราจะมาดูข้อมูลต่างๆใน Data โดยใช้ `df.describe()`")
st.image("image/NN/describe.png")
st.write("เราจะมาลองหาตัว Null และ Duplicate ของข้อมูลด้วยคำสั่ง `df.isnull().sum()` และ `df.duplicated().sum()`")
st.image("image/NN/Missing Value.png")
st.image("image/NN/Duplicates.png")
st.write("จากที่เราเห็นว่าไม่มีทั้งตัว Null และ Duplicate ใน Dataset นี้")
st.write("เราเลยไม่จำเป็นต้องแก้ไข้ dataset ในส่วนนี้แต่ที่เราจะทำเราจะเอาแค่ข้อมูลที่เป็นราคาปิดของหุ้นมาใช้ในการพยากรณ์ราคาหุ้นของ Nvidia")
st.write("โดยที่เราจะใช้โค้ด `df = df[[\"Date\", \"Close\"]]` เพื่อเลือกเฉพาะ Column ที่เราสนใจ")
st.image("image/NN/info.png")
st.write("เราจะมาเช็ค Box plot เพื่อดูว่ามี Outlier หรือไม่")
st.image("image/NN/Outlier.png")
st.write("จากที่เห็นว่าไม่มี Data Outlier ใน Dataset นี้เลยไม่จำเป็นต้องแก้ไข")

st.header("📌 **Data Preprocessing**")
st.write("เราจะทำการ Scale ข้อมูลด้วย MinMaxScaler ก่อนที่จะนำไปใช้ในการ Train Model")
st.code("""
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(df["Close"].values.reshape(-1, 1))
""", language="python")

st.write("สร้าง Sequence ของข้อมูลเป็นระยะเวลา 60 วัน")
st.code("""
# Function to create sequences
def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Create sequences of 60 days
time_steps = 60
X, y = create_sequences(prices_scaled, time_steps)
""", language="python")

st.write("แบ่งข้อมูลเป็น Training และ Testing Set โดยให้ Training Set 80% และ Testing Set 20%")
st.code("""
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
""", language="python")

st.write("ตอนนี้เราสร้าง LSTM Model ขึ้นมาแล้ว โดยที่เราจะใช้ Sequential Model ในการสร้าง Model ของเรา")
st.code("""
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(time_steps, 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(50),
    Dense(1)
])
""", language="python")

st.write("เราจะใช้ Mean Squared Error เป็น Loss Function และ Adam Optimizer ในการ Compile Model ของเรา")
st.code("""
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss="mean_squared_error")
""", language="python")

st.write("เราจะให้มีการ Stop early เพื่อไม่ให้ Overfit")
st.code("""
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
""", language="python")

st.write("เราจะ TrainModel ด้วยโค้ดดังนี้")
st.code("""
history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping])
""", language="python")

st.write("LSTM (Long Short-Term Memory) เป็นประเภทหนึ่งของ Recurrent Neural Networks (RNN) ที่ออกแบบมาเพื่อแก้ปัญหาการจำข้อมูลระยะยาว โดยมีหน่วยความจำภายในที่สามารถเก็บข้อมูลจากช่วงเวลาที่ผ่านมานานได้ ซึ่งช่วยให้สามารถเรียนรู้ลำดับข้อมูลที่มีความสัมพันธ์ระยะยาวได้ดี เช่น การทำนายราคาหุ้นจากข้อมูลในอดีต LSTM ประกอบด้วยเกตต่าง ๆ ที่ควบคุมการเก็บและลบข้อมูลที่ไม่จำเป็น, ซึ่งทำให้มันสามารถรักษาข้อมูลที่สำคัญและลืมข้อมูลที่ไม่สำคัญได้อย่างมีประสิทธิภาพ.")
st.write("ในโค้ดนี้, LSTM ใช้สำหรับการทำนายราคาหุ้น (Nvidia stock price) โดยการเรียนรู้ลำดับข้อมูลจากช่วงเวลาที่ผ่านมาเพื่อทำนายราคาที่จะเกิดขึ้นในอนาคต")

st.write("ผลลัพธ์ที่ได้จากการ Train Model ของเรา")
st.image("image/NN/predicprice.png")
st.image("image/NN/Error.png")