# build_study_guide.py
import os
import json
import re # Import the regular expression library
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# --- CONFIG ---
MY_GROQ_KEY = os.getenv("GROQ_API_KEY")
TRANSCRIPT_FOLDER = './nptel_transcripts/'
OUTPUT_FILE = 'study_guide.json'

# --- PROMPT ---
# --- PROMPT ---
summary_template = """
You are an expert NPTEL professor creating an official study guide for the 'Introduction to IoT' course.
Your task is to create a high-quality, **detailed, and comprehensive** summary of the *entire* lecture transcript provided.

The student reading this is cramming for their final exam and may not have time to watch the video. Your summary must be good enough to be their primary study material for this lecture.

**Instructions:**
1.  Start with a 2-3 sentence overview of the lecture's main topic.
2.  Create clear sections for each major sub-topic discussed. Use **Markdown headings** (e.g., `## Key Definitions`, `## CoAP Protocol Explained`).
3.  Under each heading, use **detailed bullet points** to extract:
    * All key definitions (e.g., "IoT:", "M2M:").
    * Important concepts, technical terms, and protocols (e.g., "REST Architecture", "MQTT").
    * Any comparisons made (e.g., "HTTP vs. CoAP").
    * Lists of features, characteristics, or enablers.
4.  The summary must be structured, easy to read, and cover all essential information from the transcript. Do not be "concise"; be "comprehensive".

**Transcript:**
{context}

**Comprehensive Exam Study Guide:**
"""
summary_prompt = ChatPromptTemplate.from_template(summary_template)

# --- LLM ---
llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=MY_GROQ_KEY)
summary_chain = summary_prompt | llm | StrOutputParser()

# --- SCRIPT LOGIC ---
study_guide = {}
print("Starting to build study guide...")

# --- THIS IS THE FIX ---

# 1. Get all .txt files from the folder
all_files = [f for f in os.listdir(TRANSCRIPT_FOLDER) if f.endswith(".txt")]

# 2. Define a "natural sort" key function
def get_sort_key(filename):
    """
    Extracts (week, lecture) numbers from a filename like 'W10_L46.txt'
    Returns a tuple (10, 46) for sorting.
    """
    try:
        # Use regex to find the first and second numbers
        numbers = re.findall(r'\d+', filename)
        week = int(numbers[0])
        lecture = int(numbers[1])
        return (week, lecture)
    except Exception:
        # Fallback for any filenames that don't match (e.g., 'Intro.txt')
        return (999, 999) # Put them at the end

# 3. Sort the list of files using our new key
print(f"Found {len(all_files)} files. Sorting them numerically...")
all_files.sort(key=get_sort_key)

# 4. Now, loop through the correctly sorted list
for filename in all_files:
# --- END OF FIX ---

    file_path = os.path.join(TRANSCRIPT_FOLDER, filename)
    source_name = filename.replace(".txt", "").replace("_", " ")
    print(f"Processing: {source_name}...")
    
    loader = TextLoader(file_path)
    doc = loader.load()[0]
    
    try:
        # Invoke the chain with the full document content
        summary = summary_chain.invoke({"context": doc.page_content})
        study_guide[source_name] = summary
        print(f"-> Done.")
    except Exception as e:
        print(f"-> ERROR processing {filename}: {e}")
        print("   Skipping this file.")

# --- SAVE TO FILE ---
with open(OUTPUT_FILE, 'w') as f:
    json.dump(study_guide, f, indent=4)

print(f"\nSuccess! Study guide saved to {OUTPUT_FILE}")