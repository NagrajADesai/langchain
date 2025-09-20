from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os 

load_dotenv()
token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.github.ai/inference"
model_name = "openai/text-embedding-3-small"

embedding = OpenAIEmbeddings(
    base_url=endpoint,
    model=model_name,
    api_key=token,
    dimensions=32
)

result = embedding.embed_query('Delhi is capital of India')

print(str(result))