
import requests
import pymupdf
from bs4 import BeautifulSoup
from googlesearch import search as search_function
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_search import YoutubeSearch
from typing import Union
from langchain_core.documents import Document

class url_data:
    def __init__( self , query : str  ,web_urls : Union[ list, str]= [], \
        pdfs : Union[ list, str]= [], youtube_video_ids : Union[list, str]= [] ):
        self.query = query
        self.web_urls = web_urls
        self.pdfs = pdfs
        self.youtube_video_ids = youtube_video_ids

    def get_web_url( self):
        web_url = []
        for url in search_function( self.query, num_results = 2):
            if (url == '') or (url[:8] != 'https://'):
                continue 
            web_url.append( url )

        return web_url

    def get_url_data( self,urls : Union[ list, str] ):
        if isinstance( urls , str ):
            urls = [urls]

        text = ''
        for url in urls:
            response = requests.get( url )
            soup = BeautifulSoup( response.text, 'html.parser')

            for p_teg in soup.select('p,pre'):
                if p_teg.get_text( strip = True) == '':
                    continue
                text += p_teg.get_text( strip = True )
                text += ' '


        return text

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
            for value in fetch_obj:
                language = value.language_code

            text_fetch_obj = transcript_api.fetch( video_id = video_id , languages = [language])

            for value in text_fetch_obj:
                text += value.text
                text += ' '

            text += f'{index}:{video_id} video content is finish'

        return text

    def get_video_id_for_query( self ):
        video_info_dict = YoutubeSearch( self.query, max_results = 1).to_dict()
        video_id_list = []
        for video_info in video_info_dict:
            video_id_list.append( video_info.get('id'))

        return video_id_list


    def get_all_data( self, return_all_combined_resources = False ):
        internal_urls = self.get_web_url()
        all_web_urls = internal_urls + self.web_urls
        full_url_data = self.get_url_data( urls = all_web_urls )

        pdf_data = self.get_text_from_pdf( pdfs = self.pdfs)

        internal_video_id_list = self.get_video_id_for_query()
        all_video_ids = internal_video_id_list + self.youtube_video_ids

        all_video_ids_data = self.get_text_from_youtube_video( video_ids = all_video_ids)

        docs = []
        dictionary =  {
            'web_urls_data': full_url_data,
            'pdf_data': pdf_data,
            'youtube_video_data' : all_video_ids_data
        }
        
        if return_all_combined_resources is True:
            return {'full_content' : dictionary['web_urls_data'] + ' ' + dictionary['pdf_data'] + ' ' + dictionary['youtube_video_data']}
        
        for key, value in dictionary.items():
            doc = Document(
                page_content = str(value),
                metadata = {'source': 'local'}
            )
            docs.append(doc)

        return docs

# full_data = url_data(query = 'what is llm')
# _full_data = full_data.get_all_data()
