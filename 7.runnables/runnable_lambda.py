from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
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

def word_counter(text):
    return len(text.split())

## it converts the simple function to runnable
# runnable_word_counter = RunnableLambda(word_counter)

# print(runnable_word_counter.invoke("Hi there how are you?"))

prompt = PromptTemplate(
    template="write a joke about {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_counter)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)


print(final_chain.invoke({'topic':'AI'}))