from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv
import streamlit as st
import os


# load model
load_dotenv()
token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.github.ai/inference"
model_name = "openai/gpt-4o-mini"


model = ChatOpenAI(
    base_url=endpoint,
    model=model_name,
    api_key=token
)

st.header('Reasearch Tool')

# buttons to select input 
# usr_input = st.text_input('Enter your prompt heare')
paper_input = st.selectbox("Select Reaserch Paper Name", ["Select...", "Attention is all you need", "BERT:  Pre-training of Deep Bidirectional Transformers","GPT-3: Language Models are Few-Shot Learners", "Diffusion Models GANs on Image Synthesis"])

style_input = st.selectbox("Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-oriented", "Mathematical"])

length_input = st.selectbox("Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"])

# template
template = load_prompt('3.prompts\prompt_template.json')

# fill the placeholders
prompt = template.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input
})

if st.button('Summarize'):
    result = model.invoke(prompt)
    st.write(result.content)


