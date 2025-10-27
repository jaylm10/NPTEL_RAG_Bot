🚀 NPTEL IoT Exam-Prep Assistant 🤖

Your all-in-one study companion for the NPTEL "Introduction to Internet of Things" course!

This application leverages the power of Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to help you ace your NPTEL exam. Built with Streamlit and LangChain, it provides targeted study tools based directly on the official course lecture transcripts and assignment questions.

➡️ Try the Live App Here! (Replace with your actual Streamlit app URL after deployment)

✨ Key Features

This tool is designed specifically for NPTEL students, especially those cramming in the last week!

🎯 Mock Test Simulator (50 Questions):

Takes 50 random MCQs directly from the official NPTEL weekly assignments (Weeks 1-12).

Provides instant feedback (Correct/Incorrect) and shows the official solution in a user-friendly card format.

Tracks your score and gives a final percentage with a progress bar.

Statistically, over 60% of final exam questions come from these assignments! This is your most powerful exam prep tool.

Requires no API key - completely free to use!

📖 Instant Full Course Study Guide:

A pre-generated, detailed summary of every single lecture in the course, using a high-quality LLM prompt for comprehensiveness.

Organized by week and lecture, naturally sorted (W1 L1, W1 L2 ... W2 L1 ... W10 L46...).

Uses Markdown for clear headings and bullet points within expandable sections.

Perfect for quick review or understanding topics without watching long videos.

Requires no API key - completely free!

🧠 RAG-Powered Chatbot:

Ask questions about the course content and get answers grounded specifically in the lecture transcripts.

Multiple Chat Modes:

Q&A (Default): Get direct answers based on lecture content.

Summarize Topic: Get detailed, AI-generated summaries of specific topics from the transcripts.

Explain Simply (ELI5): Get complex concepts explained in simple terms with analogies.

Quiz Me: Ask the AI to generate 3 MCQs about a topic based on the lectures.

Context Filter: Focus the chatbot's knowledge on a single lecture for hyper-specific questions!

BYOK (Bring Your Own Key): Supports both Google Gemini (via free API key) and Groq Llama 3 (via free API key) for flexibility. The app requires one of these keys for the Chatbot features, with persistent session state.

📱 Fully Responsive Design: Works great on both desktop and mobile browsers.

🛠️ Technology Stack

Frontend: Streamlit

UI Components: streamlit-option-menu

LLM Orchestration: LangChain

Vector Database: FAISS (Facebook AI Similarity Search) (CPU version)

Embeddings: Sentence Transformers (all-MiniLM-L6-v2) via langchain-huggingface

LLM APIs:

Google Gemini API (gemini-2.5-flash)

Groq API (llama-3.1-8b-instant)

Deployment: Streamlit Community Cloud

Core Language: Python

⚙️ Setup & Usage

1. Using the Deployed App:

Simply visit the live app URL: https://YOUR_STREAMLIT_APP_URL.streamlit.app/ (Replace this)

The Mock Test and Study Guide tabs work instantly without any keys.

For the Chatbot tab:

Select your preferred API provider (Google Gemini or Groq) in the sidebar.

Get a free API key from Google AI Studio or Groq Console.

Paste the key into the corresponding input box in the sidebar and click "Save Key" (or just press Enter on desktop). Your key is stored in the session state and persists until you close the tab.

Choose your Chat Mode and start asking questions!

2. Running Locally:

Clone the repository:

git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME


Create and activate a virtual environment:

python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate


Install dependencies:

pip install -r requirements.txt


(Crucial) Prepare Data: You need the course data:

Download all lecture transcripts and save them as .txt files (e.g., W1_L1_Intro.txt) inside the nptel_transcripts folder.

Download all weekly assignment PDFs. Manually extract all MCQs (question, options, correct answer key, solution) and structure them into the questions.json file following the format specified in the comments or example.

Run the vector store builder script: python build_vectorstore.py (This creates the faiss_index_iot folder).

Set the GROQ_API_KEY environment variable (e.g., using a .env file and python-dotenv - install via pip install python-dotenv). Then run the study guide builder script: python build_study_guide.py (This creates study_guide.json).

Run the Streamlit app:

streamlit run app.py


The app will open in your browser. Enter API keys in the sidebar as needed.

📁 Project Structure

NPTEL-Study-Bot/
├── faiss_index_iot/       # FAISS vector store for lecture transcripts (created by build script)
├── nptel_transcripts/     # Folder containing all raw lecture transcripts (.txt - requires manual download)
├── venv/                  # Virtual environment (ignored by Git)
├── .env                   # Local API keys for build scripts (ignored by Git)
├── .gitignore             # Specifies intentionally untracked files that Git should ignore
├── app.py                 # The main Streamlit application code
├── build_study_guide.py   # Script to generate study_guide.json (run once locally)
├── build_vectorstore.py   # Script to generate faiss_index_iot/ (run once locally)
├── questions.json         # Manually created JSON of all assignment MCQs (requires manual extraction)
├── requirements.txt       # Python dependencies for deployment
├── study_guide.json       # Pre-generated JSON summaries of all lectures (created by build script)
└── README.md              # This file


📜 License & Attribution

This project is intended for educational purposes as a study aid.

The course content (transcripts, assignment questions) belongs to NPTEL and the respective instructors. This tool is built upon publicly available NPTEL resources, typically shared under Creative Commons licenses allowing non-commercial use with attribution. Please respect NPTEL's terms of service.

The code for this application is shared under the MIT License (Consider adding an MIT License file).

Feel free to contribute or report issues on GitHub!