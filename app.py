import streamlit as st
import pickle
import os

# Set Streamlit page configuration
st.set_page_config(page_title="News Classifier", layout="wide")

# Custom CSS styling
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background-image: url('https://www.example.com/news_background.jpg');
            background-size: cover;
            background-position: center;
        }
        [data-testid="stTextArea"] textarea {
            background-color: rgba(0, 0, 0, 0.5);
            color: white;
            font-size: 16px;
            padding: 15px;
        }
        .stButton button {
            background-color: #ff6600;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            border-radius: 5px;
        }
        .stButton button:hover {
            background-color: #e65c00;
        }
    </style>
""", unsafe_allow_html=True)

# Page title and instructions
st.title("📰 News Article Classifier")
st.markdown("""
    <div style="font-size: 18px; font-weight: 300;">
        Enter a news article below, and the model will classify its category.<br>
        Get real-time predictions on the category of the news article you provide!
    </div>
""", unsafe_allow_html=True)

# Try loading model files
try:
    with open("news_classifier_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

except FileNotFoundError as e:
    st.error(f"❌ Missing file: {e.filename}")
    st.stop()
except Exception as e:
    st.error(f"❌ Failed to load model or dependencies: {e}")
    st.stop()

# Input text area
text_input = st.text_area("📝 Enter News Text", height=250, max_chars=2000)

# Classify button
if st.button("Classify"):
    if text_input.strip() == "":
        st.warning("⚠️ Please enter some news text.")
    else:
        input_vector = vectorizer.transform([text_input])
        prediction = model.predict(input_vector)
        predicted_label = label_encoder.inverse_transform([prediction[0]])[0]

        # Optional confidence if available
        if hasattr(model, "predict_proba"):
            confidence = max(model.predict_proba(input_vector)[0])
            st.success(f"📌 Predicted Category: **{predicted_label}** ({confidence*100:.2f}% confidence)")
        else:
            st.success(f"📌 Predicted Category: **{predicted_label}**")

