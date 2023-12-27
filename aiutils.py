from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from decouple import config
import sys
import os
from azureblobutil import get_bloblist

openai_api_key = config("OPENAI_API_KEY")

qa=None
agent=None

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

def get_agent():
    global agent
    if not os.path.exists('data/po1.csv'):
        print('data/po1.csv does not exist')
        return None
    if agent is not None:
        print('found agent. returning it')
        return agent
    ''' 
   agent = create_csv_agent(
    OpenAI(temperature=0),
     'data/po1.csv', verbose=False)'''
    print( f'Blob List with path::{get_bloblist()}' )
    agent = create_csv_agent(
        OpenAI(temperature=0),
        get_bloblist
        ,verbose = True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )
    print('agent is ready')
    return agent



def get_qa(): 
    global qa
    if qa is not None:
        print('found qa. returning it')
        return qa
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
    print('qa is ready')
    return qa

def query_my_question(queryText):
    is_csv_query=queryText.startswith('csv:')
    if(not is_csv_query):
        qa=get_qa()
        if qa is None:
            print('qa is none possibly due to data folder does not exist')
            return 'unable to answer your question'
        query={"query": queryText}
        print(f'querying {query}')
        result=qa.run(query)
        return result
    else:
        agent=get_agent()
        if agent is None:
            print('agent is none possibly due to data folder does not exist')
            return 'unable to answer your question'
        queryText=queryText[4:]
        result=agent.run(queryText)
        return result

# Compare this snippet from app.py:
if __name__ == '__main__':
    if len(sys.argv) <2 :
        print('not enough arguments')
        sys.exit(1)
    print(f'querying {sys.argv[1]}')
    print(query_my_question(sys.argv[1]))