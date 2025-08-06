from data import url_data 
from langchain_text_splitters.character import RecursiveCharacterTextSplitter 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate  
from langchain.memory import ConversationSummaryMemory


class generate_reponse:
    def __init__( self, model_name = 'gemini-2.5-flash' ):
        self.model_name = model_name

    def get_similar_content( self, states : dict ):
        similar_content = self.vectorstore.similarity_search(query = states['question'] )
        states['context'] = similar_content
        return states 

    def generate_content( self,states : dict ):
        context = '\n\n'.join( document.page_content for document in states['context'] )
        raw_prompt = {
                'chat_history' : self.memory.load_memory_variables({})['chat_history'],
                'context': context,
                'question': states['question']
                }

        self.prompt = self.prompt.invoke(raw_prompt)
        result = self.llm.invoke(self.prompt)
        self.memory.chat_memory.add_user_message(states['question'])
        self.memory.chat_memory.add_ai_message(result)

        return {'Answer' : result}

    def create_response( self,query):

        url_data_class = url_data(query = query )
        url_docs = url_data_class.get_all_data() 

        rc_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap = 20
        )

        splited_docs = rc_text_splitter.split_documents( url_docs )

        self.llm = init_chat_model( model = self.model_name , model_provider = 'google_genai' ,\
                google_api_key = 'AIzaSyDKUGAMTjpKpNxmVGU7Wi3pMM1QTumsYNI'
                )

        embedding = GoogleGenerativeAIEmbeddings( model = 'models/gemini-embedding-001', \
                google_api_key = 'AIzaSyDKUGAMTjpKpNxmVGU7Wi3pMM1QTumsYNI'
                )

        self.vectorstore = InMemoryVectorStore( embedding )
        self.vectorstore.add_documents(splited_docs)
        self.memory = ConversationSummaryMemory(
                memory_key = 'chat_history',
                max_num_tokens = 200,
                llm = llm,
                return_messages = True 
                )

        self.prompt = ChatPromptTemplate.from_template(
                '''
                Previous_Conversation : {chat_history}
                Context : {context}
                Question : {question}
                Answer : 
            '''
            ) 
    
    def get_response(self,query):
        
        self.create_response(query)
        states = self.get_similar_content( states = {'question' : query} )
        generated_content = self.generate_content( states )

        return {'Answer' : generated_content["Answer"].content}



generate = generate_reponse()
response = generate.get_response( 'what is llm ? ')

