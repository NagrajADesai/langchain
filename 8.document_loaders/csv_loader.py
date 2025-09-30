from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="Housing.csv")

docs = loader.load()

# for every row you will get 1 document
print(len(docs))
print(docs[0])