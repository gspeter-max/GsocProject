from data import url_data
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
import os


class generate_response:
    def __init__( self, model_name = 'gemini-2.5-flash'):

        self.llm = init_chat_model(model_name, model_provider="google-genai", \
            google_api_key = 'AIzaSyDKUGAMTjpKpNxmVGU7Wi3pMM1QTumsYNI'
            )

        self.embedding = GoogleGenerativeAIEmbeddings( model = 'models/gemini-embedding-001', \
                google_api_key = 'AIzaSyDKUGAMTjpKpNxmVGU7Wi3pMM1QTumsYNI'
                )

        self.memory = ConversationSummaryBufferMemory(
                memory_key = 'chat_history',
                return_messages = True,
                llm  = self.llm
                )

        self.prompt = ChatPromptTemplate.from_template(
                '''
                Previous_Conversation : {chat_history}
                Context : {context}
                Question : {question}
                Answer :
            '''
            )

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

        llm_prompt = self.prompt.invoke(raw_prompt).to_string()
        result = self.llm.invoke(llm_prompt)
        self.memory.chat_memory.add_user_message(states['question'])
        self.memory.chat_memory.add_ai_message(result)

        return {'Answer' : result}

    def create_response( self,url_docs):

        rc_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 20
        )

        splited_docs = rc_text_splitter.split_documents( url_docs )

        self.vectorstore = InMemoryVectorStore( self.embedding )
        self.vectorstore.add_documents(splited_docs)

    def get_response(self,query,url_docs):

        self.create_response( url_docs )
        states = self.get_similar_content( states = {'question' : query} )
        generated_content = self.generate_content( states )
        answer = generated_content['Answer'].content

        def remove_bad_things( text ):
            return re.sub( r" ```.*?\n|\n```", "", text, flags = re.DOTALL )

        return {'Answer' : remove_bad_things(answer) }


generate = generate_response()