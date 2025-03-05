import streamlit as st

st.title("📊 **การพัฒนาโมเดล SVM และ Random Forest**")

st.header("📌 **ที่มาของ Dataset**")
st.markdown("[📂 Life Expectancy (WHO) Dataset](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)")

st.header("🔹 **มี Feature 22 อัน ดังนี้**")

col1, col2 = st.columns(2)

with col1:
    st.write("- **Country** – 🌍 ประเทศที่เก็บข้อมูล")
    st.write("- **Year** – 📅 ปีที่เก็บข้อมูล")
    st.write("- **Status** – 🌍 ระดับการพัฒนาของประเทศ (**Developed / Developing**)")
    st.write("- **Life Expectancy** – ⏳ **อายุขัยเฉลี่ย (เป้าหมายที่ต้องพยากรณ์)**")
    st.write("- **Adult Mortality** – ⚰️ อัตราการเสียชีวิตของผู้ใหญ่")
    st.write("- **Infant Deaths** – 👶 จำนวนทารกที่เสียชีวิต")
    st.write("- **Under-Five Deaths** – 🧒 จำนวนเด็กอายุต่ำกว่า 5 ปีที่เสียชีวิต")
    st.write("- **HIV/AIDS** – 🦠 อัตราการเสียชีวิตจาก HIV/AIDS")
    st.write("- **Alcohol** – 🍷 การบริโภคแอลกอฮอล์เฉลี่ยต่อคน")
    st.write("- **Percentage Expenditure** – 💰 ค่าใช้จ่ายด้านสุขภาพ (**% GDP**)")
    st.write("- **Hepatitis B** – 💉 อัตราการฉีดวัคซีน Hepatitis B")
    
with col2:
    st.write("- **Measles** – 🤧 จำนวนผู้ป่วยโรคหัดต่อปี")
    st.write("- **Polio** – 💉 อัตราการฉีดวัคซีนโปลิโอ")
    st.write("- **Diphtheria** – 💉 อัตราการฉีดวัคซีนคอตีบ")
    st.write("- **Total Expenditure** – 💵 ค่าใช้จ่ายด้านสุขภาพของรัฐบาล (**% GDP**)")
    st.write("- **BMI** – ⚖️ ค่าดัชนีมวลกายเฉลี่ย")
    st.write("- **Thinness 1-19 Years** – 🏃‍♂️ อัตราภาวะผอมบางของเยาวชน (**1-19 ปี**)")
    st.write("- **Thinness 5-9 Years** – 🏃 อัตราภาวะผอมบางของเด็ก (**5-9 ปี**)")
    st.write("- **GDP** – 📈 **ผลิตภัณฑ์มวลรวมต่อหัว**")
    st.write("- **Population** – 👥 **จำนวนประชากรของประเทศ**")
    st.write("- **Income Composition of Resources** – 💳 ดัชนีการพัฒนามนุษย์ที่เกี่ยวข้องกับรายได้")
    st.write("- **Schooling** – 🎓 **จำนวนปีเฉลี่ยของการศึกษา**")


st.write("ขั้นแรกเราจะมาดู head กับ Tail ของ Dataset กันก่อน")
st.write("Head ของ Dataset")
st.image("image/ML/head.png")
st.write("Tail ของ Dataset")
st.image("image/ML/tail.png")

st.header("📌 **Data Cleaning**")
st.write("เราจะทำการทำความสะอาดข้อมูลก่อนที่จะนำไปใช้ในการเทรนโมเดล")
st.write("เราจะทำการตรวจสอบ Missing Values และทำการแก้ไขดังนี้")
st.write("1. ทำการ clean column names เผื่อในกรณีที่มีช่องว่างหรืออักขระพิเศษ")
st.write("โดยเราจะใช้คำสั่ง `df.columns = df.columns.str.strip()`")
st.write("2. ทำการหา Missing Values ใน Dataset")
st.write("โดยเราจะใช้คำสั่ง `df.isnull().sum()`")
st.image("image/ML/findnull.png")
st.write("จะเห็นได้ว่ามี Data ที่มี Missing Values อยู่หลายตัว")
st.write("จะทำการ Encode ข้อมูลที่เป็น Categorical ด้วยคำสั่ง `df['Status'] = df['Status'].map({'Developing': 0, 'Developed': 1})`")
st.write("ทำให้ได้ 2 ค่า Developing กับ Developed เป็น 0 กับ 1")
st.write("3. ทำการ Drop แถวที่มี Missing Values ใน Target Variable")
st.write("โดยเราจะใช้คำสั่ง `df = df.dropna(subset=['Life expectancy']).reset_index(drop=True)`")
st.write("4. ทำการ Handle Missing Values ด้วยการ Impute ด้วย Median")
st.write("โดยเราจะใช้คำสั่ง `df.fillna(df.median(numeric_only=True), inplace=True)`")

col3, col4 = st.columns(2)

with col3:
    st.image("image/ML/Before_null.png")
    
with col4:
    st.image("image/ML/After_null.png")

st.write("จะเห็นได้ว่าหลังจากทำการ Impute ด้วย Median แล้ว Missing Values จะหายไป")
st.write("5. ทำการ Scale Features สำหรับ SVM")
st.write("โดยเราจะใช้ MinMaxScaler ในการ Scale Features ด้วยคำสั่ง `scaler = MinMaxScaler()`")
st.write("จากนั้นทำการ Scale Features ด้วยคำสั่ง `X_scaled = scaler.fit_transform(X)`")

col5, col6 = st.columns(2)

with col5:
    st.image("image/ML/Before_distrubute.png")
    
with col6:
    st.image("image/ML/After_distrubute.png")


st.header("📌 **Model Training**")
st.write("ทำการ split Dataset โดยจะให้ Train 80% และ Test 20% ก่อนด้วยคำสั่ง:\n\n" "  `X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)`")

st.header("🔹 **Support Vector Machine (SVM)**")
st.write("โดยหลักการของ SVM คือ SVM ใช้หลักการ Hyperplane และ Margin Maximization เพื่อหาขอบเขตที่เหมาะสมที่สุดในการพยากรณ์ข้อมูล โดยใช้ Kernel Trick เช่น RBF Kernel เพื่อช่วยให้สามารถจับความสัมพันธ์เชิงซับซ้อนได้ดี โดยในโค้ด SVR (Support Vector Regression) พร้อม GridSearchCV เพื่อค้นหาค่า C และ Epsilon ที่ดีที่สุดสำหรับการพยากรณ์ Life Expectancy")
st.code("""
svm_param_grid = {'C': [0.1, 1, 5, 10], 'epsilon': [0.001, 0.01, 0.1]}
svm_grid = GridSearchCV(SVR(kernel='rbf'), svm_param_grid, cv=5, scoring='r2', n_jobs=-1)
svm_grid.fit(X_train, y_train)
best_svm_model = svm_grid.best_estimator_
""", language="python")

st.header("🔹 **Random Forest**")
st.write("โดยหลักการของ Random Forest คือ Random Forest ใช้แนวคิด Ensemble Learning โดยรวม หลาย Decision Trees เข้าด้วยกัน และใช้ Bootstrap Sampling เพื่อสุ่มข้อมูลมาสร้างต้นไม้แต่ละต้น ลด Overfitting และเพิ่มความแม่นยำ ในโค้ดของคุณใช้ GridSearchCV เพื่อค้นหาจำนวนต้นไม้ (n_estimators) และความลึกของต้นไม้ (max_depth) ที่เหมาะสมที่สุด")
st.code("""
rf_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, cv=5, scoring='r2', n_jobs=-1)
rf_grid.fit(X_train, y_train)
best_rf_model = rf_grid.best_estimator_
""", language="python")

st.header("📌 **Model Evaluation**")
st.write("เราจะทำการประเมินโมเดลทั้ง 2 โมเดล ด้วยค่า R2 Score และ Mean Absolute Error (MAE) ด้วยคำสั่ง:")
st.code("""
svm_r2 = best_svm_model.score(X_test, y_test)
rf_r2 = best_rf_model.score(X_test, y_test) 
svm_mae = mean_absolute_error(y_test, best_svm_model.predict(X_test))
rf_mae = mean_absolute_error(y_test, best_rf_model.predict(X_test))
""", language="python")
st.write("ซืึงจะได้ค่า R2 Score และ Mean Absolute Error ของ SVM และ Random Forest ออกมาเป็น")
st.write("#### 🔹 SVM Performance")
st.write("- **MAE:** 2.0701")
st.write("- **RMSE:** 3.0294")
st.write("- **R² Score:** 0.8939")

st.write("#### 🔹 Random Forest Performance")
st.write("- **MAE:** 1.0766")
st.write("- **RMSE:** 1.7281")
st.write("- **R² Score:** 0.9655")

st.image("image/ML/Compare.png")