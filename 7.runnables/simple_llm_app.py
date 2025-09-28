from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
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
    input_variables=['topic'],
    template="Suggest a catchy blog title about {topic}."
)

# define input
topic = input("Enter a topic: ")

# format the promt
formatted_prompt = prompt.format(topic=topic)

blog_title = model.invoke(formatted_prompt)

print('Generated blog title:', blog_title.content)