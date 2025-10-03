# locally hosted database using ChromaDB, to look up relevant info to pass onto model for more contextually relevant reply
from langchain_ollama.embeddings import OllamaEmbeddings # embedding model taking text and converting to vector representation (numbers)
from langchain_chroma import Chroma # vector database to store and query vector representations
from langchain_core.documents import Document # document class to represent text data with metadata
import os
import pandas as pd


df = pd.read_csv('realistic_restaurant_reviews.csv') # load the reviews dataset

embeddings = OllamaEmbeddings(model="mxbai-embed-large") # initialize the embedding model

db_location = "./chroma_langchain_db" # local directory to store the vector database
add_documents = os.path.exists(db_location) and os.listdir(db_location) # check if the directory exists and has files

if add_documents:
    documents = [] # list to hold Document objects
    ids       = [] # list to hold document IDs

    for i, row in df.iterrows(): # go row-by-row
        document = Document(
            page_content = row["Title"] + " " + row["Review"], # review text to be vectorized and looked up
            metadata     = { "rating": row["Rating"], "date": row["Date"] }, # metadata with rating and date
            id = str(i) # unique ID for each document
        )
        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    collection_name = "restaurant_reviews", # name of the collection
    persist_directory = db_location, # directory to store the database
    embedding_function = embeddings # embedding function to convert text to vectors
)

if add_documents:
    vector_store.add_documents(documents, ids=ids) # add documents to the vector store

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5} # find top 5 relevant documents and pass to LLM
    ) # retriever to fetch top 3 similar documents based on similarity search