import os
import awsgi
import threading
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from flask import Flask, render_template, request, render_template_string
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain


app = Flask(__name__)

load_dotenv()
os.environ["GOOGLE_API_KEY"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "true"

retrieval_chain = None
lock = threading.Lock()

def load_vector_db(persist_directory: Path) -> Chroma:
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=str(persist_directory), embedding_function=embedding)
    return db


def create_prompt():
    prompt = ChatPromptTemplate.from_template("""
    You are a Harry Potter expert with deep knowledge of all seven books.
    Your task is to answer the following questions based only on the provided context.
    Carefully consider all the information, and provide a concise and accurate answer.
    Format the output in proper HTML.Avoid using markdown.
    Below are multiple pieces of context that might be relevant:                                         
    <context>
    {context}
    </context>
    Question: 
    <question>                                         
    {input}
    </question>                                        
    """)
    return prompt


def initialize_retrieval_chain():
    global retrieval_chain
    if retrieval_chain is None:
        current_path = Path(__file__)
        root_path = current_path.parent
        vector_db_data_path = root_path / 'data' / 'vector_db'

        llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                    google_api_key=os.environ["GOOGLE_API_KEY"]
                )

        prompt = create_prompt()
        db = load_vector_db(persist_directory=vector_db_data_path)
        print("🔵 Loaded Vector DB with", len(db.get()['documents']), "documents.")
        retriever = db.as_retriever()

        retriever.search_kwargs['k'] = 5

        document_chain = create_stuff_documents_chain(llm, prompt)

        retrieval_chain = create_retrieval_chain(retriever, document_chain)


@app.route('/', methods=['GET', 'POST'])
def index():
    answer = ''
    if request.method == 'POST':
        initialize_retrieval_chain()
        user_input = request.form.get('input_text')
        if user_input:
            with lock:
                # 🔥 First: Test if retrieval_chain is loaded
                if retrieval_chain is None:
                    print("Retrieval chain is not initialized!")
                    answer = 'Retrieval system failed to initialize.'
                    return render_template('index.html', answer=answer)

                # 🔥 Second: Print what is retrieved
                print(f"User Input: {user_input}")

                try:
                    response = retrieval_chain.invoke({"input": user_input})
                    print("Full Response Object:", response)

                    # 🔥 Third: Check if 'answer' key is present
                    if 'answer' in response:
                        answer = response['answer']
                        if not answer.strip().startswith('<'):
                            answer = f"<p>{answer}</p>"
                    else:
                        answer = '⚠️ No answer key found in response.'
                except Exception as e:
                    print("Error during retrieval:", str(e))
                    answer = '⚠️ An error occurred during retrieval.'
        else:
            answer = 'No input provided. Please enter a question.'

    return render_template('index.html', answer=answer)


if __name__ == "__main__":
     initialize_retrieval_chain()
     app.run(debug=True, threaded=True)