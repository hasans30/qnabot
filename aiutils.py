from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, TextLoader
from decouple import config


openai_api_key = config("OPENAI_API_KEY")


# it loads a directory of documents and return vector db
def load_documents():
    loader = DirectoryLoader('data/', loader_cls=TextLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(documents)  

def get_qa():
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    texts = load_documents()
    docsearch = FAISS.from_documents(texts, embeddings)
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    # Create your Retriever
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)
    return qa

def query_my_question(queryText):
    qa=get_qa()
    query={"query": queryText}
    result=qa(query)
    return result['result']

   
# Compare this snippet from app.py:
if __name__ == '__main__':
    qa=get_qa()
    query={"query": "where am i today?"}
    result=qa(query)
    print(result['result'])
