# SHL AI Assessment Recommender

This is an AI-powered tool to recommend relevant SHL assessments based on user-entered queries such as "coding test", "sales aptitude", or "personality fit". It uses a combination of fuzzy matching, keyword similarity (TF-IDF), and filtering based on test properties like remote support, duration, and adaptiveness.

---

## 🔍 Features

- 💡 Smart assessment recommendations using fuzzy matching + keyword expansion
- ⚙️ FastAPI backend for REST API support
- 🌐 Streamlit frontend for live demo
- 📊 Filters for adaptive support, remote testing, and difficulty
- 🧠 Integrates Groq Chatbot for conversational recommendations
- 📁 Results logging to Google Sheets

---

## 🚀 Live Demo & API

| Type              | URL                                              |
|-------------------|--------------------------------------------------|
| 🌐 **Live Demo**  | [APP](https://shl-ai-assessment-recommender-2-k52c.onrender.com)     |
| 📦 **API Endpoint** | `[https://shl-ai-assessment-recommender-3-ft46.onrender.com/recommend?query=your_text]` |
| 💻 **GitHub Repository** | [GITHUB](https://github.com/priyanshur785/SHL-AI-Assessment-Recommender.git) |


---

## 📂 Project Structure
```bash
shl-ai-assessment-recommender/
├── API/
│   └── api.py                # FastAPI backend logic
├── shl_assessment.json       # Assessment catalog
├── app.py          # Streamlit frontend
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
```
---
## 📦 How to Run Locally


1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/shl-ai-assessment-recommender.git
   cd shl-ai-assessment-recommender
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the API**
   ```bash
   uvicorn API.api:app --reload
   ```

5. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

---

## 🧠 Approach Summary

We:
- Scraped SHL assessments from their official catalog
- Preprocessed the data and built search functions using:
  - Fuzzy matching
  - Optional filters like remote/adaptive/duration
- Developed both a REST API and interactive UI (Streamlit)
- Integrated Groq + Google Sheets for added intelligence and tracking

---

## 📬 Feedback & Contributions

Feel free to open issues, suggest features, or contribute code through pull requests!

---

## 🛡️ License

MIT License © 2025 Priyanshu Rai
