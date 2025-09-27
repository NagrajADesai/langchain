from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
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

parser = StrOutputParser()

# first part: identify the sentiment of feedbback
class Feedback(BaseModel):
    sentiment: Literal['positive','negative'] = Field(description='Give the sentiment of the feedback')


parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)


classifier_chain = prompt1 | model | parser2

# print(classifier_chain.invoke({'feedback':'This is the terrible smartphone'}))

# result = classifier_chain.invoke({'feedback':'This is the terrible smartphone'}).sentiment

# print(result)

##### Branching

prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback \n {feedback}",
    input_variables=['feedback']
)


prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback \n {feedback}",
    input_variables=['feedback']
)


## branch : input of tupul, which contain condition and which chain to run if true
# condition 1, chain1
# condition 2, chain2
# default chain

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x: x.sentiment =='negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)


chain = classifier_chain | branch_chain

result = chain.invoke({'feedback': 'This is a wonderfull phone'})
print(result)

# print chain 
chain.get_graph().print_ascii()