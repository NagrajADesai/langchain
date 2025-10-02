from langchain.text_splitter import RecursiveCharacterTextSplitter


# Recursive text splitter: it don't break the word, helps to keep context (split recursively-> paragraph, sentence, word)
text = """Through examination of the model architecture and source code, this technical deep dive aims to unpack MLLaMA’s architectural decisions and their implications. 
The focus is on Llama 3.2 11B vision model, which can be found at Huggingface. We’ll explore how its vision encoder differs from traditional Vision Transformers (ViT), despite similar foundations, and how it strategically preserves multi-level visual features through intermediate layer outputs. Of particular interest is its integration strategy, which uses cross-attention at specific intervals rather than the more common approaches of early fusion or late fusion.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)

result = splitter.split_text(text)

print(len(result))
print(result)
