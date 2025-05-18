import streamlit as st
import sys
import os 

sys.path.append("D:/Assignment/Capstone Project/capstone-project-Sabuna-Gamal")
from  data_utils import  prepare_features
from model import split_data , train_model, evaluate_model, save_model, cross_validate_model
from visualization import plot_actual_vs_predicted

def show(df):
    """
    Display the model training page.
    """
    st.header("📊 Model Training and Evaluation")

    # Prepare features and target
    X, y = prepare_features(df)

    # Model configuration
    test_size = st.slider("Test data size (%)", 10.0, 40.0, 20.0) / 100.0
    X_train, X_test, y_train, y_test = split_data(X, y, test_size)

    st.write(f"✅ Training Samples: {len(X_train)}")
    st.write(f"✅ Testing Samples: {len(X_test)}")

    # Choose model type
    model_type = st.selectbox(
        "🤖 Select Model Type",
        [
            "Linear Regression", "Ridge", "Lasso",
            "Random Forest", "Gradient Boosting",
            "SVM (Classification)"
        ]
    )

    task = "regression" if "Regression" in model_type or model_type in ["Ridge", "Lasso", "Random Forest", "Gradient Boosting"] else "classification"

    # Train model button
    if st.button("🚀 Train Model"):
        with st.spinner("Training in progress..."):
            model = train_model(X_train, y_train, model_type=model_type, task=task)
            metrics = evaluate_model(model, X_train, y_train, X_test, y_test, task=task)

            # Display metrics
            st.subheader("📈 Model Evaluation")
            col1, col2 = st.columns(2)

            if task == "regression":
                with col1:
                    st.metric("Train RMSE", f"{metrics['train_rmse']:.2f}")
                    st.metric("Train R²", f"{metrics['train_r2']:.4f}")
                with col2:
                    st.metric("Test RMSE", f"{metrics['test_rmse']:.2f}")
                    st.metric("Test R²", f"{metrics['test_r2']:.4f}")

                # Actual vs predicted plot
                st.subheader("📉 Actual vs Predicted (Test Set)")
                fig = plot_actual_vs_predicted(metrics['y_test'], metrics['y_pred_test'], label=model_type)
                st.pyplot(fig)

            else:  # classification
                with col1:
                    st.metric("Train Accuracy", f"{metrics['train_accuracy']:.2f}")
                    st.metric("Train F1 Score", f"{metrics['train_f1']:.2f}")
                with col2:
                    st.metric("Test Accuracy", f"{metrics['test_accuracy']:.2f}")
                    st.metric("Test F1 Score", f"{metrics['test_f1']:.2f}")

            # Cross-validation
            if st.checkbox("🔁 Show Cross-validation Score"):
                cv_score = cross_validate_model(model, X, y, task=task)
                st.write(f"Cross-Validation Score: {cv_score:.3f}")

                # Define the model storage directory 
                MODEL_DIR = "ML_Models"

                # Save model to ML_Models directory with standardized file name
                model_file_name = f"{model_type.replace(' ', '_').lower()}_model.pkl"
                save_model(model, file_name=model_file_name)
                st.success(f"✅ Model trained and saved successfully in '{MODEL_DIR}/{model_file_name}'")

            

            # Session state store
            st.session_state['model'] = model
            st.session_state['model_type'] = model_type

