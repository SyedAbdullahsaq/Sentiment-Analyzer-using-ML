import streamlit as st
import joblib

# --- Page Configuration ---
st.set_page_config(page_title="Emotion Classifier", page_icon="🧠")

# --- Load Assets ---
@st.cache_resource
def load_assets():
    model = joblib.load('logistic_regression_model.pkl')
    vectorizer = joblib.load('bow_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_assets()

# --- Emotion Mapping ---
# Mapping the array index to the emotion string and a relevant emoji
emotion_map = {
    0: ("Sadness", "😢", "#1f77b4"),  # Blue
    1: ("Anger", "😡", "#d62728"),    # Red
    2: ("Love", "🥰", "#e377c2"),     # Pink
    3: ("Surprise", "😮", "#ff7f0e"),  # Orange
    4: ("Fear", "😨", "#9467bd"),     # Purple
    5: ("Joy", "😊", "#2ca02c")       # Green
}

# --- UI ---
st.title("🧠 Emotion Sentiment Analysis")
st.write("This model classifies text into six emotional states based on your training data.")

user_text = st.text_area("What's on your mind?", placeholder="Type your feelings here...")

if st.button("Analyze Emotion"):
    if user_text.strip():
        # 1. Vectorize
        text_vector = vectorizer.transform([user_text])
        
        # 2. Predict
        prediction = model.predict(text_vector)[0]
        label, emoji, color = emotion_map[prediction]
        
        # 3. Get Confidence
        probability = model.predict_proba(text_vector).max()

        # 4. Display Result
        st.markdown("---")
        st.markdown(f"### Detected Emotion: <span style='color:{color}'>{label} {emoji}</span>", unsafe_allow_html=True)
        
        # Progress bar to show confidence
        st.write(f"Confidence: {probability:.2%}")
        st.progress(probability)
        
        # Special effects for happy emotions
        if label in ["Joy", "Love"]:
            st.balloons()
    else:
        st.warning("Please enter some text first!")

# --- Sidebar ---
with st.sidebar:
    st.header("Model Labels")
    for k, v in emotion_map.items():
        st.write(f"{k}: {v[0]} {v[1]}")