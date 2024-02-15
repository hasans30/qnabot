from doctest import OutputChecker
from langchain_openai import OpenAIEmbeddings
from langchain.agents import Tool
from langchain.tools import tool,StructuredTool,BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
# from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from decouple import config
from langchain_experimental.agents import create_pandas_dataframe_agent,create_csv_agent
from langchain_experimental.agents.agent_toolkits.pandas.prompt import PREFIX
import sys
import os
import pandas as pand
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

def get_agent(dframe=False):
    global agent
    a=[]    
    if agent is not None:
        print('found agent. returning it')
        return agent    
    if(dframe==False):
        if not os.path.exists('data/po1.csv'):
            print('data/po1.csv does not exist')
            return None
        ''' 
    agent = create_csv_agent(
        OpenAI(temperature=0),
        'data/po1.csv', verbose=False)'''
        print( f'Blob List with path::{get_bloblist()}' )
        a=get_bloblist();
        agent = create_csv_agent(
            OpenAI(temperature=0),
            a[:2]
            ,verbose = True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )
        print('agent is ready')
        return agent
    else:
        a=get_bloblist();
        if(a!=None):
            for each in a:
                if each=='data/Customer Name.csv':
                    print(f'"Check Each::{each}"')
                    df_cus = pand.read_csv(each)
                if each=='data/GL Description.csv':    
                    gls = pand.read_csv('data/GL Description.csv')
                if each=='data/Profit Center Name.csv':    
                    global pfs    
                    pfs = pand.read_csv('data/Profit Center Name.csv')
                if each=='data/Sales data_2020.csv':    
                    sale20 = pand.read_csv('data/Sales data_2020.csv')
                if each=='data/Sales data_2021.csv':    
                    sale21 = pand.read_csv('data/Sales data_2021.csv')
                if each=='data/Sales data_2022.csv':    
                    sale22 = pand.read_csv('data/Sales data_2022.csv')
                if each=='data/Sales data_2023.csv':    
                    sale23 = pand.read_csv('data/Sales data_2023.csv')
        else:

            df_cus = pand.read_csv("data/md/Customer Name.csv") 
            gls = pand.read_csv('data/md/GL Description.csv')       
            pfs = pand.read_csv('data/md/Profit Center Name.csv')
            sale20 = pand.read_csv('data/md/Sales data_2020.csv')
            sale21 = pand.read_csv('data/md/Sales data_2021.csv')
            sale22 = pand.read_csv('data/md/Sales data_2022.csv')
            sale23 = pand.read_csv('data/md/Sales data_2023.csv')
        global sales1 ;
        sales1 = pand.concat( [sale20 , sale21 , sale22 , sale23],ignore_index=True )
        print(sales1)
        # Define a list of tools
        tools = [
            Tool(
                name = "df1",
                func= mysales
                ,
                description="useful for when you need to answer questions about sales or sales amount"
            ),
            Tool(
                name = "df2",
                func=mycus
                ,
                description="useful for when you need to search for Customer name against a customer no"
            ),
            Tool(
                name = "df4",
                func=mypfs
                ,
                description="Only For Profit Center Name"
            ),
            Tool(
                name = "gls",
                func=mygls
                ,
                description="useful for when you need to search for GL description against a GL no"
            )     
        ]    
        PREFIX = ''' Always startover and do not read from memory also DO NOT exclude negative values from your calculation return the last observation as result, In Addition Consider the following:
                        "Sales" means "Amount"
                        "Month" means "Period"
                    '''
        agent = create_pandas_dataframe_agent(OpenAI(temperature=0,model='gpt-3.5-turbo-instruct',
                                                     openai_api_key=openai_api_key),
                                                     [sales1],
                                                     verbose=True,
                                                     prefix=PREFIX,
                                                     return_intermediate_steps=False,
                                                     max_iterations=20,
                                                     include_df_in_prompt=True,extra_tools=tools)
                                                    #  include_df_in_prompt=True)        
        return agent
        


def get_qa(dFrame=True): 
    global qa
    if dFrame==False:
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
    else:
        return print(f'"qa" not possible!')

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
        agent=get_agent(dframe=True)
        if agent is None:
            print('agent is none possibly due to data folder does not exist')
            return 'unable to answer your question'
        queryText=queryText[4:]
        #result=agent.run(queryText)
        result=agent.invoke(queryText)
        return result["output"]
####### >>>>Tool Sections Begin ######
@tool
def mysales(kk):
    ''' For sales related query refer this tool '''
# #     PX = '''Create dataframe as per the instruction in {kk} and refer the source from {sales1}'''
#     PX = '''Calculate the result from the supplied python kwargs, consider the input supplied in {kk}'''
#     agent2 = create_pandas_dataframe_agent(OpenAI(temperature=0,model='gpt-3.5-turbo-instruct',openai_api_key=os.environ["OPENAI_API_KEY"]),[sales1],verbose=True,include_df_in_prompt=False,return_intermediate_steps=False,max_iterations=20)
# #     return agent2.invoke('rerutn result as per the supplied prompt')
#     return agent2.invoke(' Whats the output for {df1}')
    txt = kk.replace('df1','sales1')
    return eval(txt)
    
@tool
def mycus(kk):
    ''' For Customer related information refer this'''
    return 'df_cus'
@tool
def mygls(kk):
    ''' For GL Description related information refer this'''
    return 'gls'
@tool
def mypfs(kk):
    ''' Profit Center Names'''
    return pfs['Profit Center']
####### Tool Sections End<<<<<< ######   
# Compare this snippet from app.py:
if __name__ == '__main__':
    if len(sys.argv) <2 :
        print('not enough arguments')
        sys.exit(1)
    print(f'querying {sys.argv[1]}')
    print(query_my_question(sys.argv[1]))