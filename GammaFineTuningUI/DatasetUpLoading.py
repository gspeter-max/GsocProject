import datasets 
from typing import Optional,Union
import logging

logging.getLogger().setLevel( logging.INFO )

class init_information:
    def __init__(
            self, 
            huggingface_token = None 
            ):
        self.huggingface_token = huggingface_token 

class UploadDataset( init_information ):
    def __init__(
            self , 
            path : str, 
            data_files: Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]], NoneType] = None,
            FineTuningType : str = 'ChatBotGrounding',
            split : str = None 
            ):

        super().__init__()
        if path and data_files :
            logging.info(' both path and data_files is not allow , only data_files is used in this case ') 
        if not ( path and data_files ):
            raise RuntimeError( ' no file path is given ') 

        self.path = path 
        self.data_files = data_files
        
        self.split = split  

    def load_it( self ):
        dataset = load_dataset( 
                        path = self.path, 
                        data_files = self.data_files, 
                        split = self.split 
                    )
        return dataset 

    def PrepareDataset(
            self,
            FineTuningType : str,
            dataset : Union[ IterableDataset, Dataset ] 
            ):
        if FineTunningType.lower() == 'chatbotgrounding':
            logging.info(
                """
                'make sure chatbotgrounding dataset have these columns '
                'make sure chatbotgrounding dataset have these columns '
                'make sure chatbotgrounding dataset have these columns '
                'make sure chatbotgrounding dataset have these columns '


                    { 
                    'doc' : 'Producing dairy milk is known ....' , 
                    'claim' :  'Oat milk production generally requires'
                    } 

                    or

                    {instruction}
                    ==========
                        In your answer, refer only to the context document. Do not employ any outside knowledge


                    {question}
                    ==========
                    [user request]


                    {passage 0}
                    ==========
                    [context document]

                    optional : { full conversation } [full_prompt] 

                """ 
            

            




        












