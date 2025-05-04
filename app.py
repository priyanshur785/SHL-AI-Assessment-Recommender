import streamlit as st
import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

load_dotenv()

# ----------------- Setup -----------------
st.set_page_config(page_title="SHL AI Assessment Recommender", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4F8BF9;'>🔍 SHL AI Assessment Recommender</h1>", unsafe_allow_html=True)

# ----------------- Sidebar Controls -----------------
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    top_k = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)
    max_duration = st.slider("Max duration (minutes)", min_value=10, max_value=120, value=60)
    adaptive_filter = st.checkbox("✅ Adaptive")
    remote_filter = st.checkbox("✅ Remote Support")

# ----------------- Synonym Expansion -----------------
def expand_query(query):
    synonyms = {
        "coding": "coding programming code developer",
        "remote": "remote online virtual",
        "test": "test assessment evaluation exam",
        "manager": "manager lead supervisor",
        "sales": "sales marketing business",
        "communication": "communication verbal soft skills",
        "language": "language english verbal spoken"
    }
    words = query.lower().split()
    return " ".join([synonyms.get(word, word) for word in words])

# ----------------- Load & Normalize Data -----------------
@st.cache_data
def load_data():
    try:
        with open("shl_assessments.json", "r") as f:
            data = json.load(f)
            for item in data:
                item["adaptive_support"] = str(item.get("adaptive_support", "")).lower() in ["true", "yes", "1"]
                item["remote_support"] = str(item.get("remote_support", "")).lower() in ["true", "yes", "1"]
            return data
    except FileNotFoundError:
        st.error("❌ JSON file not found.")
        return []

raw_data = load_data()
full_df = pd.DataFrame(raw_data)

# ----------------- Apply Filters -----------------
filtered_df = full_df[full_df["duration_minutes"] <= max_duration].copy()
if adaptive_filter:
    filtered_df = filtered_df[filtered_df["adaptive_support"] == True]
if remote_filter:
    filtered_df = filtered_df[filtered_df["remote_support"] == True]

# ----------------- Format Text -----------------
def format_text(row):
    return f"{row['name']} {row['test_type']} duration {row['duration_minutes']} remote {row['remote_support']} adaptive {row['adaptive_support']}"

filtered_df["formatted_text"] = filtered_df.apply(format_text, axis=1)
texts = filtered_df["formatted_text"].fillna("").astype(str)
texts = texts[texts.str.strip().str.len() > 0]

if texts.empty:
    st.error("⚠️ No data matches the current filters.")
    st.stop()

# ----------------- TF-IDF Vectorization -----------------
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(texts)

# ----------------- Recommender -----------------
def recommend(query, k=5):
    expanded = expand_query(query)
    query_vec = vectorizer.transform([expanded])
    cos_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    fuzz_scores = [fuzz.token_set_ratio(expanded, text) / 100 for text in texts]
    combined_scores = 0.7 * cos_scores + 0.3 * pd.Series(fuzz_scores)
    top_indices = combined_scores.argsort()[-k:][::-1]
    return filtered_df.iloc[top_indices], combined_scores[top_indices], fuzz_scores

# ----------------- Google Sheets Logging -----------------
def log_to_sheet(user_query, output_df):
    try:
        creds_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        sheet_name = os.getenv("GOOGLE_SHEET_NAME", "RecommendationsLog")
        if not creds_json or not os.path.exists(creds_json):
            st.warning("🟡 Google Sheets logging skipped. Missing credentials.")
            return

        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name(creds_json, scope)
        client = gspread.authorize(creds)
        sheet = client.open(sheet_name).sheet1

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for rec in output_df:
            sheet.append_row([
                now,
                user_query,
                rec["Assessment"],
                rec["URL"],
                rec["Type"],
                rec["Duration (min)"],
                rec["Remote"],
                rec["Adaptive"],
                rec["Score %"]
            ])
        st.success("✅ Recommendations logged to Google Sheets.")
    except Exception as e:
        st.warning(f"🟡 Logging failed: {e}")

# ----------------- UI Input -----------------
query = st.text_input("💬 Enter job description or keywords", placeholder="e.g. frontend developer 40 mins remote")

if query:
    results, scores, fuzz_scores = recommend(query, top_k)
    st.markdown(f"### 🎯 Top {top_k} Recommendations")
    output_df = []

    for (_, row), score, fuzz_score in zip(results.iterrows(), scores, fuzz_scores):
        with st.expander(f"🔹 {row['name']}"):
            st.markdown(f"[🔗 View Assessment]({row['url']})")
            st.markdown(f"**🧪 Test Type**: `{row['test_type']}`")
            st.markdown(f"**⏱️ Duration**: `{row['duration_minutes']} minutes`")
            st.markdown(f"**🖥️ Remote Support**: `{row['remote_support']}`")
            st.markdown(f"**⚙️ Adaptive/IRT**: `{row['adaptive_support']}`")
            st.markdown(f"**📊 Fuzzy Score**: `{round(fuzz_score * 100, 2)}%`")
            st.markdown(f"**📈 Combined Score**: `{round(score * 100, 2)}%`")

        output_df.append({
            "Assessment": row["name"],
            "URL": row["url"],
            "Type": row["test_type"],
            "Duration (min)": row["duration_minutes"],
            "Remote": row["remote_support"],
            "Adaptive": row["adaptive_support"],
            "Score %": round(score * 100, 2)
        })

    st.markdown("### 📥 Download Recommendations")
    csv_data = pd.DataFrame(output_df).to_csv(index=False).encode("utf-8")
    st.download_button("Download as CSV", csv_data, "recommendations.csv", "text/csv", use_container_width=True)

    # Log to Google Sheets
    log_to_sheet(query, output_df)
else:
    st.info("👈 Enter a query to get started.")

# ----------------- Groq Chatbot -----------------
def main():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("🚨 GROQ API key not found.")
        return

    st.title("🤖 Chat with Groq")
    st.sidebar.title('💬 Chatbot Settings')
    system_prompt = st.sidebar.text_input("System prompt:", value="Answer clearly and concisely")
    model = st.sidebar.selectbox('Model', ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it'])
    memory_len = st.sidebar.slider('Memory length', 1, 10, value=5)

    memory = ConversationBufferWindowMemory(k=memory_len, memory_key="chat_history", return_messages=True)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Ask a question")

    for message in st.session_state.chat_history:
        memory.save_context({'input': message['human']}, {'output': message['AI']})

    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)

    if user_question:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}")
        ])

        conversation = LLMChain(llm=groq_chat, prompt=prompt, memory=memory)
        response = conversation.predict(human_input=user_question)
        st.session_state.chat_history.append({"human": user_question, "AI": response})
        st.write("Chatbot:", response)

if __name__ == "__main__":
    main()

# ----------------- Footer -----------------
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Built for SHL AI Internship by Priyanshu — Powered by Streamlit</div>", unsafe_allow_html=True)
