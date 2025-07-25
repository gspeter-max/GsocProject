import datasets
from typing import Optional,Union
from types import NoneType
import logging
from huggingface_hub import login
from datasets import IterableDataset, Dataset,load_dataset

logging.getLogger().setLevel( logging.INFO )

class init_information:
    def __init__(self):
        pass

class UploadDataset( init_information ):
    def __init__(
            self ,
            path : str,
            ContextOrDocOrPassage : bool = False,
            QuestionOrClaimOrUserInput : bool = False,
            AnswerOrLabelOrResponse : bool = False,
            FineTuningType : str = 'ChatBotGrounding',
            split : Union[str,NoneType] = None
            ):

        super().__init__()
        if not path:
            raise RuntimeError(' no path is found for loading the dataset ')

        self.path = path
        self.ContextOrDocOrPassage = ContextOrDocOrPassage
        self.QuestionOrClaimOrUserInput = QuestionOrClaimOrUserInput
        self.AnswerOrLabelOrResponse = AnswerOrLabelOrResponse
        self.PossibleColumns = set([
             'context','doc','user_request', 'context_document', 'full_prompt',
             'passage','instruction','question','claim','user','answer','labels','response'
        ])
        self.FineTuningType = FineTuningType

    def load_it( self,split = 'all'):

        login(token = 'huggingface token ')
        path = self.path.split('.',maxsplit = 1)
        if len(path) <= 1:

            dataset = load_dataset(
                        path = self.path,
                        split = split
                    )
            return dataset

        if path[1] == 'csv':
            dataset = load_dataset(
                    path = 'csv',
                    data_files = self.path,
                    split = split
                    )
            print(dataset)
            return dataset

        if path[1] == 'json':
            dataset = load_dataset(
                    path = 'json',
                    data_files = self.path,
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
        data = self.load_it()
        return self.PrepareDataset(data)

    def DataArgumentation( ):
        ''' we are hold here
                1. the idea we are do that with two new features
                first using RAG ( i need to learn )
                    2. using LLM with separate

        '''
        pass



