import streamlit as st
import pandas as pd
from transformers import pipeline

# ----------------------------------
# Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="🎬 Movie Review Sentiment Analysis",
    page_icon="🎥",
    layout="wide"
)

st.title("🎬 Netflix Movie Review Sentiment Analysis")
st.markdown("Analyze whether a movie review is **Positive** or **Negative** using AI.")

# ----------------------------------
# Load Dataset
# ----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("netflix movie KGF 2.csv", sep=";")

df = load_data()

# ----------------------------------
# Load Model
# ----------------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

classifier = load_model()

# ----------------------------------
# Sidebar
# ----------------------------------
st.sidebar.header("Dataset Information")
st.sidebar.write(f"Total Reviews: **{len(df)}**")
st.sidebar.write(f"Columns: {', '.join(df.columns)}")

# ----------------------------------
# User Input
# ----------------------------------
review = st.text_area(
    "Enter a Movie Review",
    placeholder="Example: The movie was fantastic with brilliant acting..."
)

if st.button("Analyze Review"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        result = classifier(review)[0]

        sentiment = result["label"]
        confidence = result["score"] * 100

        if sentiment == "POSITIVE":
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")

        st.metric("Confidence", f"{confidence:.2f}%")

# ----------------------------------
# Dataset Preview
# ----------------------------------
st.divider()

st.subheader("Dataset Preview")
st.dataframe(df.head())

if st.checkbox("Show Full Dataset"):
    st.dataframe(df)

# ----------------------------------
# Batch Review Analysis
# ----------------------------------
st.divider()
st.subheader("Batch Sentiment Analysis")

review_column = st.selectbox(
    "Select Review Column",
    df.columns
)

if st.button("Analyze Dataset"):

    with st.spinner("Analyzing reviews..."):

        sentiments = []
        scores = []

        for review in df[review_column].astype(str):
            result = classifier(review)[0]
            sentiments.append(result["label"])
            scores.append(round(result["score"] * 100, 2))

        result_df = df.copy()
        result_df["Sentiment"] = sentiments
        result_df["Confidence (%)"] = scores

        st.success("Analysis Completed!")

        st.dataframe(result_df)

        csv = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name="sentiment_analysis_results.csv",
            mime="text/csv"
        )

  # ----------------------------------
# Sentiment Visualization
# ----------------------------------
if 'result_df' in locals():

    st.divider()
    st.subheader("Sentiment Distribution")

    sentiment_counts = result_df["Sentiment"].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        st.bar_chart(sentiment_counts)

    with col2:
        st.write("### Sentiment Count")
        st.write(sentiment_counts)

    st.write("### Sentiment Percentage")

    percentage = (
        sentiment_counts / sentiment_counts.sum() * 100
    ).round(2)

    st.dataframe(
        percentage.rename("Percentage (%)")
    )

# ----------------------------------
# About Project
# ----------------------------------
st.divider()

with st.expander("📖 About This Project"):
    st.markdown("""
### 🎬 Netflix Movie Review Sentiment Analysis

This application uses a pre-trained **Hugging Face Transformer Model**
(`distilbert-base-uncased-finetuned-sst-2-english`) to analyze movie reviews.

### Features
- ✅ Single Review Sentiment Prediction
- ✅ Batch Dataset Analysis
- ✅ Confidence Score
- ✅ Download Results as CSV
- ✅ Interactive Dashboard

### Tech Stack
- Python
- Streamlit
- Transformers (Hugging Face)
- Pandas
- PyTorch
""")

# ----------------------------------
# Footer
# ----------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:gray;'>
        <h4>🎥 Netflix Movie Review Sentiment Analysis</h4>
        <p>Built with ❤️ using Streamlit & Hugging Face Transformers</p>
    </div>
    """,
    unsafe_allow_html=True
)
