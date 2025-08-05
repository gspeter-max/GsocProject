from data import url_data 
from langchain_text_splitters.character import RecursiveCharacterTextSplitter 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chat_models import init_chat_model
from langchain import hub


url_data_class = url_data(query = 'what is llm')
url_docs = url_data_class.get_all_data() 

rc_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 20
)

splited_docs = rc_text_splitter.split_documents( url_docs )

llm = init_chat_model( model = 'gemini-2.5-flash', model_provider = 'google_genai' ,\
        google_api_key = 'AIzaSyDKUGAMTjpKpNxmVGU7Wi3pMM1QTumsYNI'
        )

embedding = GoogleGenerativeAIEmbeddings( model = 'models/gemini-embedding-001', \
        google_api_key = 'AIzaSyDKUGAMTjpKpNxmVGU7Wi3pMM1QTumsYNI'
        )

vectorstore = InMemoryVectorStore( embedding )
vectorstore.add_documents(splited_docs)
prompt = hub.pull('rlm/rag-prompt')

def get_similar_content( states : dict ):
    similar_content = vectorstore.similarity_search(query = states['question'] )
    states['context'] = similar_content
    return states 

def generate_content( prompt,states : dict ):
    context = '\n\n'.join( document.page_content for document in states['context'] )
    prompt = prompt.invoke(
        {
            'context': context,
            'question': states['question']
            }
        ).to_string() 
    result = llm.invoke(prompt) 
    return {'Answer' : result} 

states = get_similar_content( states = {'question' : 'what is llm ?'} )
generated_content = generate_content( prompt, states )

print(f'Answer : {generated_content["Answer"].content}')

