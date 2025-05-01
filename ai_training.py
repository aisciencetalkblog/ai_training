import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Trainer", layout="centered")
st.title("Multi-Input Multi-Output AI Trainer")

@st.cache_resource
def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = MLPRegressor(hidden_layer_sizes=(32, 16), activation='relu',
                         max_iter=2000, random_state=42)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    return model, scaler, r2_train, r2_test, y_test, y_test_pred, y_train, y_train_pred

# Step 1: Upload dataset
st.header("1. Upload Dataset (CSV)")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📄 Dataset Preview:")
    st.dataframe(df)

    columns = df.columns.tolist()
    st.header("2. Select Input and Output Columns")
    input_cols = st.multiselect("Select input features", columns, default=['time', 'distance'])
    output_cols = st.multiselect("Select output target(s)", columns, default=['speed'])

    if input_cols and output_cols:
        X = df[input_cols]
        y = df[output_cols]

        if st.button("🚀 Train Model"):
            model, scaler, r2_train, r2_test, y_test, y_test_pred, y_train, y_train_pred = train_model(X, y)
            st.session_state["model_ready"] = True
            st.session_state["model"] = model
            st.session_state["scaler"] = scaler
            st.session_state["input_cols"] = input_cols
            st.session_state["output_cols"] = output_cols

            # R² Scores
            st.subheader("📊 R² Score Comparison")
            st.write(f"🔍 R² on Training Set: `{r2_train:.3f}`")
            st.write(f"🔍 R² on Test Set: `{r2_test:.3f}`")

            fig_r2, ax_r2 = plt.subplots()
            ax_r2.bar(["Train", "Test"], [r2_train, r2_test])
            ax_r2.set_ylim(0, 1)
            ax_r2.set_ylabel("R² Score")
            st.pyplot(fig_r2)

            # Prediction vs Actual
            st.subheader("📈 Predicted vs Real Values (Test Set)")
            fig_pred, ax_pred = plt.subplots()
            if len(output_cols) == 1:
                ax_pred.scatter(y_test, y_test_pred, label=output_cols[0])
                min_val = min(y_test.min().values[0], y_test_pred.min())
                max_val = max(y_test.max().values[0], y_test_pred.max())
                ax_pred.plot([min_val, max_val], [min_val, max_val], 'k--')
            else:
                for i, col in enumerate(output_cols):
                    ax_pred.scatter(y_test.iloc[:, i], y_test_pred[:, i], label=col)
                min_val = min(y_test.min().min(), y_test_pred.min())
                max_val = max(y_test.max().max(), y_test_pred.max())
                ax_pred.plot([min_val, max_val], [min_val, max_val], 'k--')

            ax_pred.set_xlabel("Real Output")
            ax_pred.set_ylabel("Predicted Output")
            ax_pred.legend()
            st.pyplot(fig_pred)

# Step 4: Prediction
if st.session_state.get("model_ready", False) and len(st.session_state["output_cols"]) == 1:
    st.header("4. Predict New Output (Single Prediction)")
    new_input = []
    for col in st.session_state["input_cols"]:
        val = st.number_input(f"{col}", value=0.0, format="%.4f")
        new_input.append(val)

    if st.button("🎯 Predict"):
        model = st.session_state["model"]
        scaler = st.session_state["scaler"]
        new_input_scaled = scaler.transform([new_input])
        prediction = model.predict(new_input_scaled)[0]
        st.success(f"Predicted `{st.session_state['output_cols'][0]}`: `{prediction:.4f}`")
elif uploaded_file and len(output_cols) > 1:
    st.info("Prediction mode only works when one output column is selected.")
