from langchain_community.document_loaders import TextLoader
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
    template="write a summary for the following text - \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

## text document loader

loader = TextLoader('nvidia_driver.txt', encoding='utf-8')

docs = loader.load()

# print(docs)

# it is a list
# print(type(docs))
# print(len(docs))


# print(docs[0])

# print(type(docs[0]))

# print(docs[0].page_content)

# print(docs[0].metadata)

chain = prompt | model | parser

print(chain.invoke({'text':docs[0].page_content}))