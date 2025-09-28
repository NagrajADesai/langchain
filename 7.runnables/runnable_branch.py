from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
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


prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Summarize the following text \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()


report_gen_chain = RunnableSequence(prompt1, model, parser)

## RunnableBranch (condition, runnable)
branch_chain = RunnableBranch(
    (lambda x: len(x.split())>300, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({'topic':'russia vs ukrain'}))