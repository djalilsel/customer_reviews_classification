import joblib
import streamlit as st

st.set_page_config(page_title="Review Sentiment", page_icon="💬", layout="centered")

@st.cache_resource
def load_artifacts():
    tfidf = joblib.load("models/tfidf.joblib")
    model = joblib.load("models/clf_bal.joblib")
    return tfidf, model

tfidf, model = load_artifacts()

st.title("💬 Review Sentiment Classifier")
st.write("Paste a review below. The app predicts whether it's **Positive** or **Negative**.")

text = st.text_area("Review text", height=160, placeholder="type or paste a review...")
btn = st.button("Analyze")

if btn and text.strip():
    X = tfidf.transform([text])
    proba = model.predict_proba(X)[0, 1]
    pred = (proba >= 0.5)
    label = "🟢 Positive" if pred else "🔴 Negative"
    st.markdown(f"### {label}")
    st.caption(f"Confidence (positive probability): **{proba:.3f}**")
