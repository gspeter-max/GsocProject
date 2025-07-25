import datasets 
from typing import Optional,Union
from types import NoneType 
import logging
from datasets import IterableDataset, Dataset,load_dataset

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
            data_files: Optional[str] = None,
            ContextOrDocOrPassage : bool = False, 
            QuestionOrClaimOrUserInput : bool = False, 
            AnswerOrLabelOrResponse : bool = False,
            FineTuningType : str = 'ChatBotGrounding',
            split : Union[str,NoneType] = None 
            ):

        super().__init__()
        if path and data_files :
            logging.info(' both path and data_files is not allow , only data_files is used in this case ') 
        if not ( path and data_files ):
            raise RuntimeError( ' no file path is given ') 

        self.path = path 
        self.data_files = data_files
        self.ContextOrDocOrPassage = ContextOrDocOrPassage 
        self.QuestionOrClaimOrUserInput = QuestionOrClaimOrUserInput
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
    def DataHandling( 
                self, 
                FirstArg = '...' , 
                SecondArg = '...', 
                ThirdArg = '...', 
                FourthArg = '...'
            ):
            def is_error( arg, name ):
                if not arg:
                    raise RuntimeError( f''' make sure you spacify {name}
                        argument for chatbotgrounding 
                        ''' )

            first_arg = getattr(str,FirstArg,False)
            second_arg = getattr(str, SecondArg,False)
            third_arg = getattr(str, ThirdArg,False) 
            fourth_arg = getattr(str ,FourthArg,False)

            if not ( first_arg and second_arg and third_arg and fourth_arg ):

                raise RuntimeError( f''' make sure you spacify
                     {FirstArg} , {SecondArg} , {ThirdArg} , {FourthArg}
                     these argument for chatbotgrounding
                     ''' )
            else:
                is_error(first_arg,FirstArg) 
                is_error(second_arg,SecondArg) 
                is_error(third_arg,ThirdArg) 
                is_error(fourth_arg, FourthArg) 
        
    def PrepareDataset(
            self,
            FineTuningType : str,
            dataset : Union[ IterableDataset, Dataset ] 
            ):
        if FineTuningType.lower() == 'chatbotgrounding':
            DataHandling(
                    FirstArg = 'ContextOrDocOrPassage', 
                    SecondArg = 'QuestionOrClaimOrUserInput',
                    ThirdArg = 'AnswerOrLabelOrResponse'
                    )
            DatasetColumns = set(map(str.lower, dataset.column_names)).intersection(self.PossibleColumns) 
            return dataset.select_columns(DatasetColumns) 

