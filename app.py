import streamlit as st
import pandas as pd
from pathlib import Path
from transformers import pipeline

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Netflix Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Netflix Movie Review Sentiment Analysis")
st.write("Analyze movie reviews using a Hugging Face sentiment analysis model.")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    csv_path = Path("netflix movie KGF 2.csv")

    if not csv_path.exists():
        st.error("Dataset file not found!")
        st.stop()

    return pd.read_csv(csv_path, sep=";")


# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return pipeline(
        task="sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )


# -----------------------------
# Main
# -----------------------------
try:
    df = load_data()
    classifier = load_model()

except Exception as e:
    st.error(f"Application Error:\n\n{e}")
    st.stop()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Dataset")

st.sidebar.metric("Rows", len(df))
st.sidebar.metric("Columns", len(df.columns))

# -----------------------------
# Single Review Prediction
# -----------------------------
st.header("Analyze a Review")

review = st.text_area(
    "Enter Review",
    height=150
)

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:

        prediction = classifier(review)[0]

        label = prediction["label"]
        score = prediction["score"] * 100

        if label == "POSITIVE":
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")

        st.metric("Confidence", f"{score:.2f}%")

# -----------------------------
# Dataset Preview
# -----------------------------
st.header("Dataset Preview")

st.dataframe(df.head())

# -----------------------------
# Batch Prediction
# -----------------------------
st.header("Analyze Entire Dataset")

column = st.selectbox(
    "Select Review Column",
    df.columns
)

if st.button("Analyze Dataset"):

    with st.spinner("Analyzing..."):

        results = []

        for review in df[column].astype(str):

            pred = classifier(review)[0]

            results.append({
                "Sentiment": pred["label"],
                "Confidence": round(pred["score"] * 100, 2)
            })

        result_df = df.copy()

        result_df["Sentiment"] = [r["Sentiment"] for r in results]
        result_df["Confidence (%)"] = [r["Confidence"] for r in results]

    st.success("Analysis Complete!")

    st.dataframe(result_df)

    st.download_button(
        "Download CSV",
        result_df.to_csv(index=False),
        "sentiment_results.csv",
        "text/csv"
    )

    st.subheader("Sentiment Distribution")

    counts = result_df["Sentiment"].value_counts()

    st.bar_chart(counts)

# -----------------------------
# About
# -----------------------------
with st.expander("About Project"):

    st.write("""
This application predicts whether a movie review is Positive or Negative using a pre-trained Hugging Face Transformer model.

### Technologies Used

- Python
- Streamlit
- Transformers
- Pandas
- PyTorch
""")

st.markdown("---")
st.caption("Built using Streamlit and Hugging Face Transformers")
