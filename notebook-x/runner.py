# git clone https://github.com/gspeter-max/GsocProject.git
# cd ./GsocProject/notebook-x
# python -m runner 


from data import url_data
from generate_speech import Studio
from generate_content import generate_response
import logging 
import os 

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class create_notebook:
    def __init__( self, web_links = [], pdfs = [], youtube_video_ids = []):
        self.provided_links = (web_links, pdfs, youtube_video_ids ) 
        self.generate = generate_response()

    def send_query(self,query, generate_conversation : bool = False, \
            mp3_converted_conversation : bool = False,
            save_and_play : bool = False,
            print_here = True
        ):
        full_data = url_data(query = query, web_urls = self.provided_links[0],\
                pdfs = self.provided_links[1],
                youtube_video_ids = self.provided_links[2]
            )
        full_docs_data,full_content_dict = full_data.get_all_data( return_all_combined_resources = True )
        if generate_conversation is True:
            studio = Studio( 
                available_docs = full_docs_data
            )
            script = studio.generate_deep_deive_conversation()
            studio.generate_conversation_speech( script = script,\
                output_audio_path = 'conversation'
            )
            full_stored_path = f'{os.getcwd()}/conversation.wav'
            logger.info(f'conversation is stored in {full_stored_path}')

            if save_and_play is True:
                self.generate.play_audio(full_stored_path)
            
            if mp3_converted_conversation is True:
                mp3_stored_path = f'{os.getcwd()}/conversation.mp3'
                self.generate.convert_to_mp3( full_stored_path, mp3_stored_path)

        response = self.generate.get_response(query, full_docs_data)
        if print_here is True:
            print(response['Answer'])
        else:
            return response

notebook = create_notebook()
notebook.send_query(query = 'what is llm ? ', mp3_converted_conversation = True, generate_conversation = True)