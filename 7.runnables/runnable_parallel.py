from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel
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

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a Linkedin post about {topic}',
    input_variables=['topic']
)


parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkedin': RunnableSequence(prompt2, model, parser)
})

output = parallel_chain.invoke({'topic':'AI'})

print(output)