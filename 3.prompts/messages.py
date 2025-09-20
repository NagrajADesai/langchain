from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
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

# mentain history of messages
messages = [
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about LangChain')
]

result = model.invoke(messages)
# append the answer by ai to messages
messages.append(AIMessage(content=result.content))

print(messages)