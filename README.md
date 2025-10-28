# 🚀 NPTEL IoT Exam-Prep Assistant 🤖

Your **all-in-one study companion** for the NPTEL _"Introduction to Internet of Things"_ course!

This application leverages the power of **Large Language Models (LLMs)** and **Retrieval-Augmented Generation (RAG)** to help you **ace your NPTEL exam**.  
Built with **Streamlit** and **LangChain**, it provides targeted study tools based directly on the official course lecture transcripts and assignment questions.

---

## 🌐 Try the Live App

➡️ **[Launch the App](https://nptel-iot-prep.streamlit.app/)**  
*(Replace the above link with your actual Streamlit app URL after deployment.)*

---

## ✨ Key Features

This tool is designed specifically for NPTEL students — especially those **cramming in the last week!** 😄

---

### 🎯 Mock Test Simulator (50 Questions)

- Takes **50 random MCQs** directly from the official NPTEL weekly assignments (Weeks 1–12).  
- Provides **instant feedback** (Correct/Incorrect) and shows the **official solution** in a user-friendly card format.  
- Tracks your **score** and gives a **final percentage with a progress bar**.  
- **Fact:** Over 60% of final exam questions come from these assignments!  
- Requires **no API key** – completely **free to use**!

---

### 📖 Instant Full Course Study Guide

- A **pre-generated, detailed summary** of every single lecture in the course using a high-quality LLM prompt.  
- Organized by **week and lecture**, naturally sorted (`W1L1`, `W1L2`, `W2L1`, ...).  
- Uses **Markdown** for clear headings and bullet points within expandable sections.  
- Perfect for **quick review** or understanding topics without watching long videos.  
- Requires **no API key** – completely **free!**

---

### 🧠 RAG-Powered Chatbot

Ask questions about the course content and get **answers grounded directly in the lecture transcripts.**

#### 💬 Chat Modes

- **Q&A (Default):** Get direct answers from the course lectures.  
- **Summarize Topic:** Generate detailed summaries of any topic.  
- **Explain Simply (ELI5):** Understand complex concepts with simple analogies.  
- **Quiz Me:** Generate 3 custom MCQs about a topic.  

#### 🎯 Context Filter
Focus the chatbot’s knowledge on a **specific lecture** for hyper-accurate responses.

#### 🔑 BYOK (Bring Your Own Key)
Supports:
- **Google Gemini API** (`gemini-2.5-flash`)
- **Groq API** (`llama-3.1-8b-instant`)

Simply paste your free API key into the sidebar — stored safely in your session state.

---

### 📱 Fully Responsive Design
Works great on both **desktop and mobile browsers**.

---

## 🛠️ Technology Stack

| Component | Technology |
|------------|-------------|
| **Frontend** | Streamlit |
| **UI Components** | `streamlit-option-menu` |
| **LLM Orchestration** | LangChain |
| **Vector Database** | FAISS (CPU version) |
| **Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2`) via `langchain-huggingface` |
| **LLM APIs** | Google Gemini API, Groq API |
| **Deployment** | Streamlit Community Cloud |
| **Core Language** | Python |

---

## ⚙️ Setup & Usage

### 🔹 Using the Deployed App

1. Visit the live app:  
   👉 [https://YOUR_STREAMLIT_APP_URL.streamlit.app/](https://nptel-iot-prep.streamlit.app/)  

2. The **Mock Test** and **Study Guide** tabs work instantly — **no API keys required!**

3. For the **Chatbot tab**:
   - Select your preferred API provider (**Gemini** or **Groq**) in the sidebar.  
   - Get a **free API key** from:
     - [Google AI Studio](https://aistudio.google.com)
     - [Groq Console](https://console.groq.com)
   - Paste the key into the input box and click **"Save Key"**.  
   - Choose your Chat Mode and start asking questions!

---

### 🔹 Running Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME

# 2. Create and activate a virtual environment
python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# Prepare the Data
# Download all lecture transcripts and save as .txt files inside the nptel_transcripts/ folder.
# Example: W1_L1_Intro.txt, W1_L2_History.txt, etc.

# Download all weekly assignment PDFs and manually extract MCQs into questions.json with the required format.

# 4. Build the FAISS vector store:
python build_vectorstore.py

# 5. Set your GROQ_API_KEY (via .env file using python-dotenv) and generate the study guide:
python build_study_guide.py

# 6. Run the Streamlit app:
streamlit run app.py

# Open your browser, enter your API key (if using the chatbot), and enjoy!

```

## 📜 License & Attribution

This project is intended **for educational purposes** as a study aid.

- The **course content** (transcripts, assignments) belongs to **NPTEL** and respective instructors.
  
The **code** for this project is shared under the **MIT License**.

---

## 🤝 Contributing

Contributions are welcome! 🎉  

You can:
- 🐞 **Report bugs**
- 💡 **Suggest new features**
- 🚀 **Submit pull requests**

---

## 💬 Acknowledgements

Thanks to:
- 🎓 **NPTEL & IITs** for open educational content  
- ⚙️ **LangChain**, **FAISS**, and **Streamlit** for enabling easy AI app development  
- 🤖 **Google Gemini** and **Groq** for providing powerful free APIs  

---

**Built with ❤️ to help students prepare smarter**

