from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader


loader = DirectoryLoader(
    path = 'books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

## load() = load all document once

docs = loader.load()


print(len(docs))

print(docs[130].page_content)
print(docs[130].metadata)


## lazy_load() = it creates a generator (one document at a time)
# docs = loader.lazy_load()

# for document in docs:
#     print(document.metadata)