from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template= "Write a detailed report on {topic}",
    input_variables=['topic']
)


# 2st prompt -> detailed report
template2 = PromptTemplate(
    template= "Write a 5 line summary on the following text. /n {text}",
    input_variables=['text']
)
 

# parser
parser = StrOutputParser()

# parset takes the only text input which is equal to result.content
chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)
