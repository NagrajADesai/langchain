from langchain_openai import OpenAI, ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.github.ai/inference"
model_name = "openai/gpt-4o-mini"


llm = ChatOpenAI(
    base_url=endpoint,
    model=model_name,
    api_key=token
)

result = llm.invoke('What is the financial capital of India')

print(result.content)