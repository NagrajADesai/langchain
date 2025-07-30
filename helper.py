import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ['GOOGLE_API_KEY'] =  GOOGLE_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
MODEL = "gpt-4o-mini"

# extract text from the pdf files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# create chunks of text
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    
    return vector_store


def get_conversational_chain(vector_store):
    # gemini model
    # llm = ChatGoogleGenerativeAI(
    # model_name="gemini-1.5-flash-latest",
    # temperature=0.7,
    # max_output_tokens=500,
    # convert_system_message_to_human=True  #Crucial for properly working with system messages
    # )
    
    # openai model
    llm = ChatOpenAI(
        temperature=0.7,
        openai_api_base="https://models.inference.ai.azure.com",
        model_name=MODEL
    )    

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=vector_store.as_retriever(),
        memory=memory,
        verbose=False
    )
    
    return conversation_chain