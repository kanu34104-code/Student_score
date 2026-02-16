import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Page configuration
st.set_page_config(page_title="Student Score SVM Classifier", layout="wide")

st.title("🎓 Student Career & Job Predictor (SVM)")
st.write("Yeh app SVM model ka use karke predict karti hai ki student ke paas part-time job hai ya nahi.")

# 1. Data Loading
@st.cache_data
def load_data():
    df = pd.read_csv(r'student-scores .csv')
    # Basic Cleaning
    df['gender'] = df['gender'].str.capitalize()
    return df

df = load_data()

# Sidebar for user inputs
st.sidebar.header("Model Settings")
test_size = st.sidebar.slider("Test Data Size (%)", 10, 50, 20) / 100
kernel_type = st.sidebar.selectbox("SVM Kernel", ("linear", "poly", "rbf", "sigmoid"))

# 2. Preprocessing
features = ['absence_days', 'weekly_self_study_hours', 'math_score', 
            'history_score', 'physics_score', 'chemistry_score', 
            'biology_score', 'english_score', 'geography_score']
target = 'part_time_job'

X = df[features]
y = LabelEncoder().fit_transform(df[target])

# Scaling and Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Training
svm_model = SVC(kernel=kernel_type)
svm_model.fit(X_train_scaled, y_train)
y_pred = svm_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# 4. Display Results
col1, col2 = st.columns(2)

with col1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

with col2:
    st.subheader("Model Performance")
    st.metric(label="Accuracy Score", value=f"{accuracy * 100:.2f}%")
    st.write(f"Used Kernel: **{kernel_type}**")

# 5. Manual Prediction Tool
st.divider()
st.subheader("🔍 Check Single Student Prediction")
input_data = {}

cols = st.columns(3)
for i, feature in enumerate(features):
    with cols[i % 3]:
        input_data[feature] = st.number_input(f"Enter {feature}", value=int(df[feature].mean()))

if st.button("Predict Job Status"):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = svm_model.predict(input_scaled)
    
    if prediction[0] == 1:
        st.success("Result: Is student ke paas **Part-time Job** hone ke chances hain!")
    else:

        st.warning("Result: Is student ke paas **Job nahi** hone ke chances hain.")
