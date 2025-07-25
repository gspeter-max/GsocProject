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
            ContextOrDocOrPassage : bool = None, 
            QuestionOrClaimOrUserInput : bool = None, 
            AnswerOrLabelOrResponse = bool = None,
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
        self.ContextOrDocOrPassage = ContextOrClaimOrUserInput 
        self.QuestionOrClaimOrUser = QuestionOrClaimOrUser
        self.AnswerOrLabelOrResponse = AnswerOrLabelOrResponse
        self.split = split
        self.PossibleColumns = set([
             'context','doc', 'passage','question','claim','user','answer','label','response'
        ]) 

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
            if not ( self.ContextOrDocOrPassage and self.QuestionOrClaimOrUser and self.AnswerOrLabelOrResponse ):
                raise RuntimeError( ''' make sure you spacify 
                     ContextOrDocOrPassage and QuestionOrClaimOrUser and AnswerOrLabelOrResponse
                     these argument for chatbotgrounding 
                     ''' )
            else: 
                if not self.ContextOrDocOrPassage:
                    raise RuntimeError( ''' make sure you spacify 
                         ContextOrDocOrPassage 
                        argument for chatbotgrounding 
                        ''' )
                if not self.QuestionOrClaimOrUser:
                    raise RuntimeError( ''' make sure you spacify 
                         QuestionOrClaimOrUser
                        argument for chatbotgrounding 
                        ''' )
                if not self.AnswerOrLabelOrResponse:
                    raise RuntimeError( ''' make sure you spacify 
                         AnswerOrLabelOrResponse
                        argument for chatbotgrounding 
                        ''' )
            DatasetColumns = set(map(str.lower, dataset.column_names)).intersection(self.PossibleColumns) 
            'hold on here  and the logic is take only these columns from dataset and then tokenize it and done it ' 


 

            

            




        












