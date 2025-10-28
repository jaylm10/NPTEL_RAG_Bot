import streamlit as st
import os
import time
import json
import random
import re
from streamlit_option_menu import option_menu

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser


try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    st.error("Please install langchain-huggingface: pip install -U langchain-huggingface")
    st.stop()

# --- 1. App Configuration ---

st.set_page_config(page_title="NPTEL IoT Study Bot", layout="wide")

# --- 2. Caching & Loading Models (Heavy Operations) ---
@st.cache_resource
def load_embeddings():
    """Load the embedding model (cached)"""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_vector_store(_embeddings):
    """Load the local FAISS vector store (cached)"""
    VECTORSTORE_PATH = 'faiss_index_iot'
    if not os.path.exists(VECTORSTORE_PATH):
        st.error(f"Vector store not found at '{VECTORSTORE_PATH}'. Please run `build_vectorstore.py` first.")
        return None
    
    return FAISS.load_local(
        VECTORSTORE_PATH, 
        _embeddings, 
        allow_dangerous_deserialization=True
    )

# --- 3. Load Pre-Generated Data (Light Operations) ---
@st.cache_data
def load_study_guide():
    """Load the pre-generated JSON study guide"""
    try:
        with open('study_guide.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"Error": "study_guide.json not found. Run build_study_guide.py first!"}

@st.cache_data
def load_questions():
    """Load the pre-generated JSON mock test questions"""
    try:
        with open('questions.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"Error": "questions.json not found. Please create it manually from the PDFs."}

study_guide_data = load_study_guide()
question_data = load_questions()

# --- 4. Main App Layout (Tabs) ---

# Define the options for the menu
menu_options = ["Mock Test", "Chatbot", "Full Study Guide"]
menu_icons = ["patch-check-fill", "chat-dots-fill", "book-half"]

# Initialize selected_tab in session state if it doesn't exist
if "selected_tab" not in st.session_state:
    st.session_state.selected_tab = menu_options[0] # Default to the first option ("Mock Test")

# Determine the default index based on the current session state
try:
    # Find the index of the currently selected tab name
    default_tab_index = menu_options.index(st.session_state.selected_tab)
except ValueError:
    default_tab_index = 0 # Fallback to the first option if state is invalid

# Create the option menu, explicitly setting the default index
current_selection = option_menu(
    menu_title=None,
    options=menu_options,
    icons=menu_icons,
    orientation="horizontal",
    key="main_menu",
    default_index=default_tab_index, # Explicitly set the selected tab visual
    styles={ # Your corrected styles
        "container": {"padding": "5px !important", "background-color": "#fafafa"},
        "icon": {"color": "#636AF2", "font-size": "20px"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin":"0px",
            "--hover-color": "#eee",
            "padding": "10px",
            "color": "#333333",
            "background-color": "transparent",
        },
        "nav-link-selected": {
            "background-color": "#636AF2",
            "color": "white",
            "font-weight": "bold",
        },
    }
)

# --- Crucial: Update session state AFTER getting the selection ---
# If the user clicked a different tab, update the session state variable
# This ensures the default_index is correct on the *next* rerun
if st.session_state.selected_tab != current_selection:
    st.session_state.selected_tab = current_selection
    # Rerun immediately to reflect the change and ensure the correct tab content shows
    st.rerun() 

# Use the session state variable for controlling which tab content to display
selected_tab = st.session_state.selected_tab 

# --- 5. Persistent Sidebar ---
with st.sidebar:
    st.header("Configuration")
    
    provider = st.radio(
        "Choose API Provider:",
        ("Google Gemini", "Groq Llama 3"),
        help="Select which API to use. The Chatbot tab requires an API key."
    )
    
    # --- API Key Help Expander ---
    with st.expander("❓ How to get a Free API Key?"):
        if provider == "Google Gemini":
            st.markdown(
                """
                1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
                2. Sign in with your Google account.
                3. Click **"Get API key"** > **"Create API key"**.
                4. **Copy** the key & paste below.
                """, unsafe_allow_html=True
            )
            #  Placeholder for image
        elif provider == "Groq Llama 3":
            st.markdown(
                """
                1. Go to [Groq Console](https://console.groq.com/keys).
                2. Sign up (Free).
                3. Click **"API Keys"** > **"Create API Key"**.
                4. Name it (e.g., "nptel-bot") & **Create**.
                5. **Copy** the key & paste below.
                """, unsafe_allow_html=True
            )
            #  Placeholder for image
    # --- END Expander ---
    
    # Initialize Session State for API Keys
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = ""

    # --- API Key Input Form (Request 1) ---
    # Using a form helps mobile users "submit" their key
    with st.form(key="api_key_form"):
        if provider == "Google Gemini":
            st.text_input("Enter your Google API Key:", type="password", key="api_key_input")
        elif provider == "Groq Llama 3":
            st.text_input("Enter your Groq API Key:", type="password", key="groq_api_key_input")
        
        # This button is helpful for mobile users
        submit_button = st.form_submit_button(label="Save Key")

    # When form is submitted, update the session state
    if submit_button:
        if provider == "Google Gemini":
            st.session_state.api_key = st.session_state.api_key_input
        elif provider == "Groq Llama 3":
            st.session_state.groq_api_key = st.session_state.groq_api_key_input
        st.success("Key Saved!")

    # --- End of API Key Form ---

    st.markdown("---")

    # --- Chatbot-specific sidebar items ---
    if selected_tab == "Chatbot":
        st.header("Chat Mode")
        chat_mode = st.radio(
            "Choose an action for your next message:",
            ("Q&A (Default)", "Summarize Topic", "Explain Simply (ELI5)", "Quiz Me")
        )
        
        st.header("Context Filter")
        lecture_list = ["Search All Lectures"]
        if "Error" not in study_guide_data:
            # Sort the lecture list for the dropdown
            def get_sort_key_for_list(lecture_name):
                numbers = re.findall(r'\d+', lecture_name)
                if len(numbers) >= 2:
                    return (int(numbers[0]), int(numbers[1]))
                elif len(numbers) == 1:
                    return (int(numbers[0]), 0)
                else:
                    return (999, 999)
            lecture_list = ["Search All Lectures"] + sorted(study_guide_data.keys(), key=get_sort_key_for_list)
        
        selected_lecture = st.selectbox(
            "Focus on a specific lecture:",
            options=lecture_list,
            help="Focus the chatbot on a single lecture to get specific answers."
        )
        if selected_lecture != "Search All Lectures":
            st.success(f"Mode: Searching only {selected_lecture}")
    
    else:
        # Set defaults when not on the Chatbot tab
        chat_mode = "Q&A (Default)"
        selected_lecture = "Search All Lectures"

    st.markdown("---")
    st.info(
        "Content sourced from the NPTEL 'Introduction to IoT' course. "
        "This is a non-commercial study tool."
    )


# --- TAB 1: MOCK TEST (Request 3: 50 Questions) ---
if selected_tab == "Mock Test":
    st.title("⏱️ NPTEL Mock Test")
    st.info("Test your knowledge! This is a **50-question** mock test, randomly selected from all 12 weekly assignments.")

    if "Error" in question_data:
        st.error(question_data["Error"])
    else:
        # Initialize test state in session
        if 'test_questions' not in st.session_state:
            st.session_state.test_questions = []
            st.session_state.current_q_index = 0
            st.session_state.user_score = 0
            st.session_state.show_answer = False
            st.session_state.answered_current_q = False
            st.session_state.last_answer_correct = False

        # --- Test Start Screen ---
        if not st.session_state.test_questions:
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                if st.button("Start 50-Question Mock Test", use_container_width=True, type="primary"):
                    if len(question_data) < 50:
                        st.error(f"Error: Not enough questions in questions.json. Need at least 50. Found: {len(question_data)}")
                    else:
                        st.session_state.test_questions = random.sample(question_data, 50)
                        st.session_state.current_q_index = 0
                        st.session_state.user_score = 0
                        st.session_state.show_answer = False
                        st.session_state.answered_current_q = False
                        st.rerun()

        # --- Test in Progress Screen ---
        elif st.session_state.current_q_index < 50: # Changed to 50
            question = st.session_state.test_questions[st.session_state.current_q_index]
            q_num = st.session_state.current_q_index + 1

            st.progress(q_num / 50, text=f"Question {q_num} of 50") # Changed to 50
            
            with st.container(border=True):
                st.subheader(f"Question {q_num}")
                st.markdown(f"**Score: {st.session_state.user_score} / {q_num - 1}**")
                st.markdown("---")
                
                with st.form(key=f"q_form_{q_num}"):
                    st.markdown(f"### {question['question_text']}")
                    
                    options = [f"{k}. {v}" for k, v in question['options'].items()]
                    
                    user_answer = st.radio(
                        "Choose your answer:", 
                        options, 
                        key=f"q_radio_{q_num}",
                        index=None  # No default selection
                    )
                    
                    check_button = st.form_submit_button("Check Answer")

                    if check_button:
                        st.session_state.show_answer = True
                        
                        if not st.session_state.answered_current_q:
                            if user_answer is None:
                                st.session_state.last_answer_correct = False
                            else:
                                correct_text = question['correct_text']
                                if user_answer == correct_text:
                                    st.session_state.user_score += 1
                                    st.session_state.last_answer_correct = True
                                else:
                                    st.session_state.last_answer_correct = False
                            
                            st.session_state.answered_current_q = True

            # --- Show Feedback AFTER form is submitted ---
            if st.session_state.show_answer:
                if st.session_state.last_answer_correct:
                    st.success("🎉 Correct! Great job.", icon="🎉")
                else:
                    correct_text = question['correct_text']
                    st.error(f"❌ Incorrect. The correct answer was: **{correct_text}**", icon="❌")
                
                st.info(f"**Solution:** {question['solution']}")

                col1, col2, col3 = st.columns([1,2,1])
                with col2:
                    if st.button("Next Question", use_container_width=True, type="primary"):
                        st.session_state.current_q_index += 1
                        st.session_state.show_answer = False
                        st.session_state.answered_current_q = False
                        st.rerun()

        # --- Test Finished Screen ---
        else:
            st.success(f"**Test Complete!**")
            percentage = (st.session_state.user_score / 50) * 100 # Changed to 50
            
            st.metric(
                label="Final Score", 
                value=f"{st.session_state.user_score} / 50", # Changed to 50
                delta=f"{percentage:.2f}%"
            )
            
            if percentage >= 60:
                st.balloons()
                st.markdown("### Great job! You're in a strong position for the exam. 🚀")
            else:
                st.warning("### Good effort. Keep reviewing the Study Guide and your transcripts!")

            if st.button("Start New Test"):
                st.session_state.test_questions = [] # Clear the session state to restart
                st.rerun()

# --- TAB 2: CHATBOT ---
if selected_tab == "Chatbot":
    st.title("🚀 NPTEL IoT Exam-Prep Assistant")

    # --- Initialize Chat History ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Display Past Chat Messages ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                st.caption(f"Sources: {', '.join(message['sources'])}")

    # --- Load Models & Define Chains ---
    llm = None 
    retriever = None
    rag_chain = None
    summary_chain = None
    quiz_chain = None
    eli5_chain = None 

    # Read keys from st.session_state
    if provider == "Google Gemini":
        if not st.session_state.api_key:
            st.sidebar.warning("Please enter your Google API Key to use Gemini.")
        else:
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=st.session_state.api_key)
                st.sidebar.success("Using Google Gemini API.")
            except Exception as e:
                st.sidebar.error(f"Google API Error: {e}")

    elif provider == "Groq Llama 3":
        if not st.session_state.groq_api_key:
            st.sidebar.warning("Please enter your Groq API Key to use Groq.")
        else:
            try:
                llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3, groq_api_key=st.session_state.groq_api_key)
                st.sidebar.success("Using Groq (Llama 3) API.")
            except Exception as e:
                st.sidebar.error(f"Groq API Error: {e}")

    # --- If an LLM was successfully created, load the vector store and chains ---
    if llm:
        try:
            embeddings = load_embeddings()
            vectorstore = load_vector_store(embeddings)
            
            if vectorstore:
                # Create retriever with or without filter
                search_kwargs = {"k": 5}
                if selected_lecture != "Search All Lectures":
                    search_kwargs = {"k": 5, "filter": {"source": selected_lecture}}
                
                retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                # --- DEFINE ALL 4 CHAINS ---

                # Chain 1: Q&A (RAG)
                qa_template = """
                You are an NPTEL teaching assistant for 'Introduction to IoT'.
                Answer the user's question based *only* on the provided context.
                Important Rule: If the provided context does NOT contain relevant information...
                "I'm sorry, but I couldn't find information on that topic in the transcripts."
                Context: {context}
                Question: {question}
                Answer:
                """
                qa_prompt = ChatPromptTemplate.from_template(qa_template)
                rag_chain = (
                    RunnableParallel(context=retriever, question=RunnablePassthrough()) | 
                    {"answer": ({"context": (lambda x: format_docs(x["context"])), "question": (lambda x: x["question"])} | qa_prompt | llm | StrOutputParser()),
                     "sources": (lambda x: list(set(doc.metadata.get("source", "Unknown") for doc in x["context"])))}
                )

                # Chain 2: Summarization (Detailed Prompt)
                summary_template = """
                You are an expert NPTEL professor. Your task is to create a detailed, in-depth summary
                of the lecture transcript provided, based on the user's topic: **{topic}**.
                Your summary MUST be comprehensive. Start with a brief overview, then provide detailed
                bullet points for all key topics, definitions, important concepts, and any processes
                or architectures mentioned. This is for a student who is studying for an exam.
                **Important Rule:** If the provided context does NOT contain relevant information...
                "I'm sorry, but I couldn't find information on that topic in the transcripts."
                **Context:** {context}
                **Detailed Summary (following the rule above):**
                """
                summary_prompt = ChatPromptTemplate.from_template(summary_template)
                summary_chain = ({"context": retriever | format_docs, "topic": RunnablePassthrough()} | summary_prompt | llm | StrOutputParser())

                # Chain 3: Explain Simply (ELI5)
                eli5_template = """
                You are a friendly and patient teacher. Your task is to explain a complex
                NPTEL concept to a student who is panicking for an exam.
                Use simple analogies and plain English. Avoid jargon. Be concise and clear.
                Important Rule: If the provided context does NOT contain relevant information...
                "I'm sorry, but I couldn't find information on that topic to explain."
                Context: {context}
                Topic to Explain: {topic}
                Simple Explanation:
                """
                eli5_prompt = ChatPromptTemplate.from_template(eli5_template)
                eli5_chain = ({"context": retriever | format_docs, "topic": RunnablePassthrough()} | eli5_prompt | llm | StrOutputParser())

                # Chain 4: Quiz Generator
                quiz_template = """
                You are an NPTEL professor. First, review the provided context.
                Your task is to generate 3 multiple-choice-questions (MCQs) to test a student
                on this topic: **{topic}**.
                Important Rule: If the context does NOT contain relevant information...
                "I'm sorry, but I couldn't find enough information on that topic to create a quiz."
                Context: {context}
                Questions (following the rule above):
                """
                quiz_prompt = ChatPromptTemplate.from_template(quiz_template)
                quiz_chain = ({"context": retriever | format_docs, "topic": RunnablePassthrough()} | quiz_prompt | llm | StrOutputParser())

        except Exception as e:
            st.error(f"Error initializing models: {e}")

    # --- The Chat Input (at the bottom) ---
    if prompt := st.chat_input("Ask about your NPTEL course..."):
        
        if not llm:
            st.warning("Please select a provider and enter a valid API key in the sidebar to start.")
            st.stop()
        
        if not rag_chain or not summary_chain or not quiz_chain or not eli5_chain:
            st.error("Chains could not be loaded. Did the vector store load correctly?")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_content = ""
            sources = []
            
            with st.spinner("Thinking..."):
                if chat_mode == "Summarize Topic":
                    response_content = summary_chain.invoke(prompt)
                
                elif chat_mode == "Explain Simply (ELI5)":
                    response_content = eli5_chain.invoke(prompt)
                
                elif chat_mode == "Quiz Me":
                    response_content = quiz_chain.invoke(prompt)
                
                else: # Default is "Q&A (Default)"
                    response = rag_chain.invoke(prompt)
                    if isinstance(response, dict):
                        response_content = response.get("answer", "No answer found.")
                        sources = response.get("sources", [])
                    else:
                        response_content = "I'm sorry, but I couldn't find information on that topic in the transcripts."

            # "Streaming" effect
            placeholder = st.empty()
            full_response = ""
            for chunk in response_content.split():
                full_response += chunk + " "
                time.sleep(0.01)
                placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
            
            placeholder.markdown(response_content, unsafe_allow_html=True)
            
            if sources:
                st.caption(f"Sources: {', '.join(sources)}")
            
            bot_message = {"role": "assistant", "content": response_content}
            if sources:
                bot_message["sources"] = sources
            st.session_state.messages.append(bot_message)

# --- TAB 3: STUDY GUIDE (Request 2: Sorted) ---
if selected_tab == "Full Study Guide":
    st.title("📖 Full Course Study Guide")
    st.info("This is an instant, pre-generated summary of every lecture. Perfect for cramming! No API key required.")
    
    if "Error" in study_guide_data:
        st.error(study_guide_data["Error"])
    else:
        try:
            # --- NATURAL SORT FIX (Request 2) ---
            def get_sort_key(item_tuple):
                # item_tuple[0] is the key, e.g., "W1 L1 Intro"
                numbers = re.findall(r'\d+', item_tuple[0])
                if len(numbers) >= 2:
                    return (int(numbers[0]), int(numbers[1]))
                elif len(numbers) == 1:
                    return (int(numbers[0]), 0)
                else:
                    return (999, 999) # Fallback for non-standard names
            
            sorted_lectures = sorted(study_guide_data.items(), key=get_sort_key)
            # --- END OF FIX ---
            
            for source_name, summary in sorted_lectures:
                with st.expander(f"**{source_name}**"):
                    st.markdown(summary)
                    
        except Exception as e:
            st.error(f"Error displaying study guide: {e}")
            st.error("Is the 'study_guide.json' file in the correct format? (e.g., 'W1 L1 Intro')")