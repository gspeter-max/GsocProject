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
            path : str = 'qualifire/grounding-benchmark',
            ContextOrDocOrPassage : bool = False,
            QuestionOrClaimOrUserInput : bool = False,
            AnswerOrLabelOrResponse : bool = False,
            FineTuningType : str = 'ChatBotGrounding'
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

    def load_it( self,split = 'all', hf_token : str = None):

        login(token = hf_token)
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
                FirstArgName = '...' ,
                SecondArgName = '...',
                ThirdArgName = '...',
                FourthArgName = '...'
            ):

            def is_error( arg, name ):
                if not name == '...':
                    if not arg:
                        raise RuntimeError( f''' make sure you spacify {name}
                            argument for chatbotgrounding
                            ''' )

            first_arg = getattr(self, FirstArgName, False)
            second_arg = getattr(self, SecondArgName, False)
            third_arg = getattr(self, ThirdArgName, False)
            fourth_arg = getattr(self, FourthArgName, False)

            true_list = [first_arg,second_arg,third_arg,fourth_arg]

            if true_list.count(False) > true_list.count(True): # (x > 1) is Flase means one argument is not available \
            # we are just comparing to true the logic you know 

                raise RuntimeError( f''' make sure you spacify
                     {FirstArgName} , {SecondArgName} , {ThirdArgName} , {FourthArgName}
                     these argument for chatbotgrounding
                     ''' )
            else:
                is_error(first_arg, FirstArgName)
                is_error(second_arg, SecondArgName)
                is_error(third_arg, ThirdArgName)
                is_error(fourth_arg, FourthArgName)

    def PrepareDataset(
            self,
            dataset : Union[ IterableDataset, Dataset ]
            ):
        if self.FineTuningType.lower() == 'chatbotgrounding':
            self.DataHandling(
                    FirstArgName = 'ContextOrDocOrPassage',
                    SecondArgName = 'QuestionOrClaimOrUserInput',
                    ThirdArgName = 'AnswerOrLabelOrResponse'
                    )
            DatasetColumns = set(map(str.lower, dataset.column_names)).intersection(self.PossibleColumns)
            return dataset.select_columns(DatasetColumns)

    def __call__(self, hf_token):
        logging.warning(f'''make sure your dataset look like this

        dataset =
            'train' : 'features' : ['feature1','feature2'],
                       'num_rows' : '2873' ,
            'val' : 'features' : ['feature1','feature2'],
                       'num_rows' : '2873',
        
        If your dataset's columns do not match the required names, 
        use a function like data.rename_column() to align them.


        ''')
        data = self.load_it(hf_token)
        return self.PrepareDataset(data)

    def DataArgumentation(self):
        ''' we are hold here
                1. the idea we are do that with two new features
                first using RAG ( i need to learn )
                    2. using LLM with separate

        '''
        pass
#Datasets = UploadDataset(
#     ContextOrDocOrPassage = True,
#     QuestionOrClaimOrUserInput = True,
#     AnswerOrLabelOrResponse = True
# )
# dataset = Datasets()

