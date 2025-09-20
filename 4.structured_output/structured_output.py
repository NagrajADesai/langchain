from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
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

## define structure of dictionary
class Review(TypedDict):

    key_themes: Annotated[list[str], "write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal['positive', 'negative', 'neutral'], "Return sentiment of the review either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]], "write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "write down all the cons inside a list"]


structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.""")

print(result)