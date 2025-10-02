from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


## Normal text splitter
# text = """Through examination of the model architecture and source code, this technical deep dive aims to unpack MLLaMA’s architectural decisions and their implications. The focus is on Llama 3.2 11B vision model, which can be found at Huggingface. We’ll explore how its vision encoder differs from traditional Vision Transformers (ViT), despite similar foundations, and how it strategically preserves multi-level visual features through intermediate layer outputs. Of particular interest is its integration strategy, which uses cross-attention at specific intervals rather than the more common approaches of early fusion or late fusion.
# """

# splitter = CharacterTextSplitter(
#     chunk_size=100,
#     chunk_overlap=0,
#     separator=''
# )

# result = splitter.split_text(text)

# print(result)

## import pdf and then  split

loader = PyPDFLoader('..\\8.document_loaders\\mixture-of-recursions-paper.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

result = splitter.split_documents(docs)

print(result[0])