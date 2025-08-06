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

    def create_response( self,query):

        url_data_class = url_data(query = query )
        url_docs = url_data_class.get_all_data() 

        rc_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap = 20
        )

        splited_docs = rc_text_splitter.split_documents( url_docs )

        llm = init_chat_model( model = self.model_name , model_provider = 'google_genai' ,\
                google_api_key = 'AIzaSyDKUGAMTjpKpNxmVGU7Wi3pMM1QTumsYNI'
                )

        embedding = GoogleGenerativeAIEmbeddings( model = 'models/gemini-embedding-001', \
                google_api_key = 'AIzaSyDKUGAMTjpKpNxmVGU7Wi3pMM1QTumsYNI'
                )

        vectorstore = InMemoryVectorStore( embedding )
        vectorstore.add_documents(splited_docs)
        memory = ConversationSummaryMemory(
                memory_key = 'chat_history',
                max_num_tokens = 200,
                llm = llm,
                return_messages = True 
                )

        prompt = ChatPromptTemplate.from_template(
                '''
                Previous_Conversation : {chat_history}
                Context : {context}
                Question : {question}
                Answer : 
            '''
            ) 

        def get_similar_content( states : dict ):
            similar_content = vectorstore.similarity_search(query = states['question'] )
            states['context'] = similar_content
            return states 

        def generate_content( prompt,memory,states : dict ):
            context = '\n\n'.join( document.page_content for document in states['context'] )
            raw_prompt = {
                    'chat_history' : memory.load_memory_variables({})['chat_history'],
                    'context': context,
                    'question': states['question']
                    }

            prompt = prompt.invoke(raw_prompt)
            result = llm.invoke(prompt)
            memory.chat_memory.add_user_message(states['question'])
            memory.chat_memory.add_ai_message(result)

            return {'Answer' : result}
    
    def get_response(self,query):
        
        states = get_similar_content( states = {'question' : query} )
        generated_content = generate_content( prompt, states )

        print(f'Answer : {generated_content["Answer"].content}')
