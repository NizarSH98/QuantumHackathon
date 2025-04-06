import streamlit as st
import pandas as pd
from utils import clean_and_prepare_data
from models.classical_model import train_classical
from models.quantum_model import train_quantum
from joblib import dump
import io
import matplotlib.pyplot as plt
from PIL import Image

# Optional: if you're still using it
#from quantum_utils import load_quantum_model

st.set_page_config(page_title="Cancer Detection App", layout="wide")

# Load and display the logo
logo = Image.open("assests\WhatsApp Image 2025-04-05 at 4.48.19 PM.jpeg")  # fix typo in path
col1, col2 = st.columns([1, 5])
with col1:
    st.image(logo, width=500)
with col2:
    st.title("🧬 Cancer Detection: Classical vs Quantum ML")

uploaded_file = st.file_uploader("📤 Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Raw Data Preview")
    st.dataframe(df.head())

    model_type = st.radio("🤖 Choose model type", ["Classical ML", "Quantum ML"])

    with st.spinner("🔄 Preprocessing and training..."):
        try:
            X_train, X_test, y_train, y_test = clean_and_prepare_data(df)

            if model_type == "Classical ML":
                metrics = train_classical(X_train, X_test, y_train, y_test)

                st.subheader("📊 Regression Metrics")
                for key, val in metrics.items():
                    st.markdown(f"**{key}:** {val}")

            elif model_type == "Quantum ML":
                progress_bar = st.progress(0)
                max_iterations = 20  # Match with COBYLA maxiter

                metrics, elapsed, objective_vals, weights, scaler, pca = train_quantum(
                    X_train, X_test, y_train, y_test,
                    on_iteration=lambda it: progress_bar.progress(it / max_iterations)
                )

                st.success(f"⚛️ Quantum VQC Accuracy: {metrics['Accuracy']:.2%}")
                st.info(f"🕒 Training Time: {elapsed:.2f} seconds")

                st.subheader("📊 Classification Metrics")
                for key, val in metrics.items():
                    if isinstance(val, dict):
                        st.markdown(f"**{key}:**")
                        st.json(val)
                    elif isinstance(val, list):
                        st.markdown(f"**{key}:**")
                        st.write(val)
                    else:
                        st.markdown(f"**{key}:** {val}")

                # Save quantum model
                buffer = io.BytesIO()
                dump({'weights': weights, 'scaler': scaler, 'pca': pca}, buffer)
                buffer.seek(0)
                st.download_button(
                    label="Download Quantum Model",
                    data=buffer,
                    file_name="quantum_model.joblib",
                    mime="application/octet-stream"
                )

                if objective_vals:
                    st.subheader("📉 Objective Function Progress")
                    fig, ax = plt.subplots()
                    ax.plot(range(len(objective_vals)), objective_vals, marker='o')
                    ax.set_title("Objective Function Value vs. Iteration")
                    ax.set_xlabel("Iteration")
                    ax.set_ylabel("Objective Function Value")
                    ax.grid(True)
                    st.pyplot(fig)

            st.balloons()

        except Exception as e:
            st.error(f"❌ Error: {e}")

else:
    st.info("Upload a CSV file to get started.")
