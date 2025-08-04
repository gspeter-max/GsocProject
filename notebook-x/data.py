import requests 
import pymupdf 
from bs4 import BeautifulSoup 
from googlesearch import search as search_function
from youtube_transcript_api import YouTubeTranscripApi
from youtube_search import YoutubeSearch

class url_data:
    def __init__( self , query : str  , urls : Union[ list, str], pdfs : Union[ list, str]):
        self.query = query 
        self.urls = urls 
        self.pdfs = pdfs

    def get_url_data( self, dataset : dict):

        for url in search_function( self.query, num = 1, stop = 1):
            response = requests.get(url) 
            soup = BeautifulSoup( response.text, 'html.parser')
            text = [] 
            for p_teg in soup.select('p'):
                if p_teg.get_text( strip = true) == '':
                    continue 
                text.append( p_teg.get_text( strip = true )) 
        
            dataset['content'] = text 
            dataset['title'] = soup.title.get_text(strip = true)

        return dataset 

    def get_user_url_data( self,urls : Union[ list, str] ):
        if isinstance( urls , str ):
            urls = [urls] 

        dataset = {} 
        for url in urls: 
            response = requests.get( url )
            soup = BeautifulSoup( response.text, 'html.parser')
            text = [] 
            
            for p_teg in soup.select('p'):
                if p_teg.get_text( strip = true) == '':
                    continue 
                text.append( p_teg.get_text( strip = true )) 
        
            dataset['content'] = text 
            dataset['title'] = soup.title.get_text(strip = true)

        return dataset
    
    def get_text_from_pdf( self , pdfs : Union[ list, str ] ):
        if isinstance( pdfs, str):
            pdfs = [pdfs]

        text = '' 
        for index, pdf in enumerate(pdfs):
            for pages in pymupdf(pdf):
                text += pages.get_text()
            
            text += f'{index} pdf is end'

        return text

    def get_text_from_youtube_video( self , video_ids : Union[ list, str ] ):
        if isinstance( video_ids, str ):
            video_ids = [video_ids] 
        
        transcript_api = YouTubeTranscriptApi()
        text = ''
        for index, video_id in enumerate(video_ids):
            fetch_obj = transcript_api.list( video_id = video_id )
            for value in fetch_object:
                language = value.language_code
            
            text_fetch_obj = transcript_api.fetch( video_id = video_id , languages = [language])
            
            for value in text_fetch_obj:
                text += value.text 
                text += ' '

            text += f'{index}:{video_id} video content is finish'
        
        return text 

    def get_video_id_for_query( self ):
        video_info_dict = YoutubeSearch( self.query, max_results = 10).to_dict()
        video_id_list = [] 
        for video_info in video_info_dict:
            video_id_list.append( video_info.get('id'))

        return video_id_list 

    


        
        



