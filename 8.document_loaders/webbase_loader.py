from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
token = os.environ["GITHUB_TOKEN"]

endpoint = "https://models.github.ai/inference"
model_name = "openai/gpt-4o-mini"

model = ChatOpenAI(
    base_url=endpoint,
    model=model_name,
    api_key=token
)

prompt = PromptTemplate(
    template="Answer the following question \n {question} from the following text \n {text}",
    input_variables=['question','text']
)

parser = StrOutputParser()


## web based loader
url = "https://www.flipkart.com/apple-macbook-air-m2-8-gb-256-gb-ssd-mac-os-monterey-mlxw3hn-a/p/itmc2732c112aeb1?pid=COMGFB2GSG8EQXCQ&lid=LSTCOMGFB2GSG8EQXCQJWHH2F&marketplace=FLIPKART&cmpid=content_computer_8965229628_gmc"


loader = WebBaseLoader(url)

docs = loader.load()

# print(len(docs))

# print(docs[0].page_content)

chain = prompt | model | parser

result = chain.invoke({'question':"give the specifications of the device", 'text':docs[0].page_content})

print(result)