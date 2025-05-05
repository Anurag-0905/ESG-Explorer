import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline
import matplotlib.pyplot as plt
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from collections import Counter
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import streamlit as st

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
 
nltk.download('stopwords')
nltk.download('punkt')

st.set_page_config(page_title="ESG Explorer", layout="wide")

st.sidebar.title("ESG Explorer")
st.sidebar.info(
    """
    **ESG Explorer** is an interactive NLP application for analyzing
    corporate sustainability (ESG) narratives. The app supports sentiment analysis,
    topic modeling using LDA, and interactive topic visualization with pyLDAvis.
    Adjust the settings, supply your ESG narratives, and receive actionable insights.
    """
)

st.title("ESG Explorer: Interactive NLP Insights for Corporate Sustainability Narratives")
st.header("Input ESG Narratives")

input_method = st.radio("Choose Input Method:", ("Text Area", "Upload File"))

if input_method == "Text Area":
    user_input = st.text_area(
        "Enter ESG narratives (each narrative on a new line):",
        """Company A is committed to a sustainable future. Over the past year, it has reduced its carbon emissions by 25% through the adoption of renewable energy sources and increased energy efficiencies.
At Company B, social responsibility is at the core of our operations. Our sustainability report highlights key initiatives, including fair labor practices, diversity and inclusion.
Company C has overhauled its governance framework to ensure ethical conduct and transparency.
""")
    narratives = [line for line in user_input.split("\n") if line.strip()]
else:
    uploaded_file = st.file_uploader("Upload your ESG narratives text file", type=["txt"])
    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8")
        narratives = [line for line in content.split("\n") if line.strip()]
    else:
        narratives = []

if not narratives:
    st.warning("No ESG narratives provided yet. Please enter text or upload a file.")
else:
    st.success(f"{len(narratives)} ESG narrative(s) loaded.")

analysis_option = st.selectbox("Select Analysis Type:", ["Sentiment Analysis", "Topic Modeling", "Both"])

if analysis_option in ["Sentiment Analysis", "Both"]:
    st.header("Sentiment Analysis")
    if st.button("Run Sentiment Analysis"):
        sentiment_pipeline = pipeline("sentiment-analysis")
        sentiment_results = []
        for report in narratives:
            result = sentiment_pipeline(report)[0]
            sentiment_results.append(result)
        st.subheader("Individual Sentiment Results")
        for idx, res in enumerate(sentiment_results, start=1):
            st.write(f"Report {idx}: {res}")
        positive_scores = [res['score'] if res['label'] == 'POSITIVE' else 1 - res['score'] 
                           for res in sentiment_results]
        labels = [f"Report {i+1}" for i in range(len(sentiment_results))]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, positive_scores, color='green')
        ax.set_title("Positive Sentiment Scores")
        ax.set_xlabel("ESG Report")
        ax.set_ylabel("Positive Score")
        ax.set_ylim(0, 1)
        st.pyplot(fig)
        overall_score = sum(positive_scores) / len(positive_scores)
        st.write(f"**Overall Average Positive Score:** {overall_score:.2f}")

if analysis_option in ["Topic Modeling", "Both"]:
    st.header("Topic Modeling and Visualization")
    if st.button("Run Topic Modeling"):
        stop_words = set(stopwords.words('english'))
        def preprocess_text(text):
            tokens = word_tokenize(text.lower())
            tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
            return tokens

        processed_texts = [preprocess_text(report) for report in narratives]
        if processed_texts:
            dictionary = corpora.Dictionary(processed_texts)
            corpus = [dictionary.doc2bow(text) for text in processed_texts]
            num_topics = st.slider("Select Number of Topics:", min_value=2, max_value=10, value=3, step=1)
            lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, random_state=42)
            st.write("### LDA Topics (Top 10 words each):")
            topics = lda_model.print_topics(num_words=10)
            for topic in topics:
                st.write(f"**Topic {topic[0]}:** {topic[1]}")

            coherence_model = CoherenceModel(model=lda_model, texts=processed_texts, dictionary=dictionary, coherence='c_v')
            coherence_score = coherence_model.get_coherence()
            st.write(f"**Topic Coherence Score:** {coherence_score:.3f}")
            
            dominant_topics = []
            for bow in corpus:
                topic_distribution = lda_model.get_document_topics(bow)
                dominant_topic = max(topic_distribution, key=lambda x: x[1])[0] if topic_distribution else -1
                dominant_topics.append(dominant_topic)
            topic_counts = Counter(dominant_topics)
            topics_list, counts = list(topic_counts.keys()), list(topic_counts.values())
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.bar([f"Topic {t}" for t in topics_list], counts, color='blue')
            ax2.set_title("Dominant Topic Distribution")
            ax2.set_xlabel("Topic")
            ax2.set_ylabel("Frequency")
            st.pyplot(fig2)
            
            vis = gensimvis.prepare(lda_model, corpus, dictionary)
            html_string = pyLDAvis.prepared_data_to_html(vis)
            st.subheader("Visualization")
            st.components.v1.html(html_string, height=800, scrolling=True)
        else:
            st.error("No processed narratives available for topic modeling.")
