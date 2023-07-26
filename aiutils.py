from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from decouple import config
import sys
import os

openai_api_key = config("OPENAI_API_KEY")


# it loads a directory of documents and return vector db
def load_documents():
    documents=[]
    if not os.path.exists('data/'):
        print('data folder does not exist')
        return None
    loader_map = {
                '**/*.txt': TextLoader, 
                '**/*.pdf': PyPDFLoader
    }

    for extension,doc_loader_class in loader_map.items():
        loader = DirectoryLoader('data/', glob=extension, loader_cls=doc_loader_class)
        documents.extend(loader.load())
            
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(documents)  

def get_qa():
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    texts = load_documents()
    if texts is None:
        print('texts is none possibly due to data folder does not exist')
        return None
    docsearch = FAISS.from_documents(texts, embeddings)
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    # Create your Retriever
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

                    Context: {context}
                    
                    Question: {question}

                    """
    PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=False, chain_type_kwargs=chain_type_kwargs)
    return qa

def query_my_question(queryText):
    qa=get_qa()
    if qa is None:
        print('qa is none possibly due to data folder does not exist')
        return 'unable to answer your question'
    query={"query": queryText}
    result=qa.run(query)
    return result

# Compare this snippet from app.py:
if __name__ == '__main__':
    if len(sys.argv) <2 :
        print('not enough arguments')
        sys.exit(1)
    print(f'querying {sys.argv[1]}')
    print(query_my_question(sys.argv[1]))