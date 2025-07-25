import datasets
from typing import Optional,Union
from types import NoneType
import logging
from huggingface_hub import login
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
        if not path :
            if not data_files :
                raise RuntimeError( ' no file path is given ')

        self.path = path
        self.data_files = data_files
        self.ContextOrDocOrPassage = ContextOrDocOrPassage
        self.QuestionOrClaimOrUserInput = QuestionOrClaimOrUserInput
        self.AnswerOrLabelOrResponse = AnswerOrLabelOrResponse
        self.PossibleColumns = set([
             'context','doc','user_request', 'context_document', 'full_prompt', 'passage','question','claim','user','answer','label','response'
        ])
        self.FineTuningType = FineTuningType

    def load_it( self,split):
        login(token = 'huggingface token ')
        dataset = load_dataset(
                        path = self.path,
                        data_files = self.data_files,
                        split = split
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
                if not name == '...':
                    if not arg:
                        raise RuntimeError( f''' make sure you spacify {name}
                            argument for chatbotgrounding
                            ''' )

            first_arg = getattr(self,FirstArg,False)
            second_arg = getattr(self,SecondArg,False)
            third_arg = getattr(self,ThirdArg,False)
            fourth_arg = getattr(self,FourthArg,False)

            true_list = [first_arg,second_arg,third_arg,fourth_arg]

            if true_list.count(False) > true_list.count(True):

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
            dataset : Union[ IterableDataset, Dataset ]
            ):
        if self.FineTuningType.lower() == 'chatbotgrounding':
            self.DataHandling(
                    FirstArg = 'ContextOrDocOrPassage',
                    SecondArg = 'QuestionOrClaimOrUserInput',
                    ThirdArg = 'AnswerOrLabelOrResponse'
                    )
            DatasetColumns = set(map(str.lower, dataset.column_names)).intersection(self.PossibleColumns)
            return dataset.select_columns(DatasetColumns)

    def __call__(self):
        logging.warning(f'''make sure your dataset look like this

        dataset =
            'train' : 'features' : ['feature1','feature2'],
                       'num_rows' : '2873' ,
            'val' : 'features' : ['feature1','feature2'],
                       'num_rows' : '2873',

        if your dataset columns is not match to { self.PossibleColumns }
        if not have using data.rename_columns or another function is exist to rename columns

        ''')
        data = self.load_it(split = 'general')
        return self.PrepareDataset(data)

