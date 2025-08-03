from datasets import load_dataset
from typing import Optional,Union
from types import NoneType
import logging
from huggingface_hub import login
from transformers import AutoTokenizer

logger = logging.getLogger()
logger.setLevel( logging.INFO )

class init_information:
    def __init__(self):
        pass


class InvalidDatasetFormatError(Exception):
    def __init__(self, message="Dataset does not match expected input-output format."):
        super().__init__(message)


class UploadDataset( init_information ):
    def __init__( self ,hf_token : str ,tokenizer,path : Union[str, None] = None, \
        FineTuningType : str = None ,max_length = None):
        super().__init__()
        
        logger.info('max_length is "None",update to it max_length = 1024 ')
        self.path = path
        self.FineTuningType = FineTuningType
        self.tokenizer = tokenizer
        self.hf_token = hf_token
        self.max_length = max_length if max_length is not None else 1024 

    def load_it( self):

        login(token = self.hf_token)
        if self.path is None:
            return None 

        path = self.path.split('.',maxsplit = 1)
        if len(path) <= 1:

            dataset = load_dataset(
                        path = self.path,
                    )
            return dataset

        if path[1] == 'csv':
            dataset = load_dataset(
                    path = 'csv',
                    data_files = self.path
                    )
            return dataset

        if path[1] == 'json':
            dataset = load_dataset(
                    path = 'json',
                    data_files = self.path
                    )
            return dataset

    def on_loading(self):
        dataset = self.load_it() 
        
        if self.FineTuningType.lower() == 'instruction_fine_tuning':
            if dataset is None:
                dataset = load_dataset('yahma/alpaca-cleaned')
                dataset['train'] = dataset['train'].select(range(10))

            if ('train' in list(dataset.keys())) and \
                (sorted(dataset['train'].column_names) == sorted(['instruction','input','output'])):

                self.tokenizer.add_special_tokens({'sep_token': '[sep]'})
                def map_function(example):
                    text = [f'instucation : {instrunct} input : {inputs} output : [sep] {output}' for  instrunct, inputs,output in zip(example['instruction'], example['input'], example['output'])]
                    text = self.tokenizer(text, truncation = True, padding = 'max_length',max_length = self.max_length ,return_tensors = 'pt')
                    sep_token_id = self.tokenizer.convert_tokens_to_ids('[sep]')
                    label = text.input_ids.clone()
                    index = (label == sep_token_id).nonzero(as_tuple = True)
                    for row, col in zip(index[0],index[1]):
                        label[row, :col] = -100
                    text['labels'] = label
                    return text 
                
                train_dataset = dataset['train'].map( map_function, batched = True, remove_columns = dataset['train'].column_names)
                eval_dataset = dataset['train'].map( map_function, batched = True, remove_columns = dataset['train'].column_names) if 'eval' in list(dataset.keys()) else None 

                return {
                    'train_dataset' : train_dataset, 
                    'eval_dataset' : eval_dataset
                }

            else:
                raise InvalidDatasetFormatError('''
                    Dataset does not match expected input-output format
                    dataset = {
                        train :  {
                            "instruction": "Translate to French: 'Good morning'",
                            "input": "",
                            "output": "Bonjour"
                        }, 
                        eval : {
                            "instruction": "Translate to French: 'Good morning'",
                            "input": "",
                            "output": "Bonjour"
                        } 
                    }
                    ''')
        if self.FineTuningType.lower() == 'code_generation':

            if dataset is None:
                dataset= load_dataset('glaiveai/glaive-code-assistant-v2')
                dataset = dataset.rename_columns({
                    "question" : 'prompt',
                    'answer' : 'completion'
                })
                dataset['train'] = dataset['train'].select(range(1000))

            if ('train' in list(dataset.keys())) and \
                (sorted(dataset['train'].column_names) == sorted(['prompt','completion'])):
                
                def map_function(example):

                    text = [f'prompt : {prompt} completion : {completion}' for prompt, completion in zip(example['prompt'], example['completion'])]
                    text = self.tokenizer(text, truncation = True, padding = 'max_length', max_length = self.max_length,return_tensors = 'pt')
                    sep_token_id = self.tokenizer.convert_tokens_to_ids('[sep]')
                    label = text.input_ids.clone()
                    index = (label == sep_token_id).nonzero(as_tuple = True)
                    for row, col in zip(index[0],index[1]):
                        label[row, :col] = -100
                    text['labels'] = label
                    return text 
                
                train_dataset = dataset['train'].map( map_function, batched = True, remove_columns = dataset['train'].column_names)
                eval_dataset = dataset['train'].map( map_function, batched = True, remove_columns = dataset['train'].column_names) if 'eval' in list(dataset.keys()) else None 

                return {
                    'train_dataset' : train_dataset, 
                    'eval_dataset' : eval_dataset
                }

            else:
                raise InvalidDatasetFormatError('''
                Dataset does not match expected input-output format
                    dataset = {
                        train : { "prompt": "Write Python code for...", "completion": "def foo(): ..." }, 
                        eval : { "prompt": "Write Python code for...", "completion": "def foo(): ..." }
                    }
                ''')

        if self.FineTuningType.lower() == 'chat_fine_tuning':
            if dataset is None:
                dataset = load_dataset('Crystalcareai/Code-feedback-sharegpt-renamed')
                dataset = dataset.remove_columns('id')

            if ('train' in list(dataset.keys())) and (sorted(dataset['train'].column_names) == sorted(['messages'])) and (sorted(dataset['train']['messages'][1][1]) == sorted(['role','value'])):

                    def map_function(example):
                        outer_list = []
                        for list_example in example['messages']: 
                            inner_list = []
                            for _dict in list_example:
                                inner_list.append(f'role : {_dict["role"]} content : {_dict["value"]}')
                            inner_list = ' '.join(inner_list)
                            outer_list.append(inner_list)
                            
                        text = self.tokenizer(outer_list, truncation = True, padding = 'max_length',max_length = self.max_length ,return_tensors = 'pt')
                        sep_token_id = self.tokenizer.convert_tokens_to_ids('[sep]')
                        label = text.input_ids.clone()
                        index = (label == sep_token_id).nonzero(as_tuple = True)
                        for row, col in zip(index[0],index[1]):
                            label[row, :col] = -100
                        text['labels'] = label
                        return text 
                    
                    train_dataset = dataset['train'].map( map_function, batched = True, remove_columns = dataset['train'].column_names)
                    eval_dataset = dataset['train'].map( map_function, batched = True, remove_columns = dataset['train'].column_names) if 'eval' in list(dataset.keys()) else None 

                    return {
                        'train_dataset' : train_dataset, 
                        'eval_dataset' : eval_dataset
                    }

            else :
                raise InvalidDatasetFormatError('''
                        Dataset does not match expected input-output format
                        dataset = {
                            train : { "messages": [ {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."} ] }, 
                            eval : { "messages": [ {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."} ] }
                        }
                ''')
        
        if self.FineTuningType.lower() in  ('question_answering','rag_fine_tuning'):
            if dataset is None:

                if self.FineTuningType.lower() == 'question_answering':
                    dataset = load_dataset('mou3az/Question-Answering-Generation-Choices')
                    dataset = dataset.remove_columns(['distractors'])
                    dataset['train'] = dataset['train'].select(range(1000))
                
                if self.FineTuningType.lower() == 'rag_fine_tuning':
                    
                    dataset = load_dataset('microsoft/ms_marco','v2.1')
                    dataset['train'] = dataset['train'].rename_columns({
                        'passages' : 'context',
                        'query' : 'question',
                        'answers' : 'answer'
                    })

                    dataset = dataset.remove_columns(['query_id','query_type','wellFormedAnswers'])
                    dataset['train'] = dataset['train'].select(range(1000))

            if ('train' in list(dataset.keys())) and \
                (sorted(dataset['train'].column_names) == sorted(['question','context','answer'])):
                
                def map_function(example):
                    
                    text = [f'context : {context} question : {question} answer : [sep] {answer}' for  context, question,answer in zip(example['context'], example['question'], example['answer'])]
                    text = self.tokenizer(text, truncation = True, padding = 'max_length',max_length = self.max_length, return_tensors = 'pt')
                    sep_token_id = self.tokenizer.convert_tokens_to_ids('[sep]')
                    label = text.input_ids.clone()
                    index = (label == sep_token_id).nonzero(as_tuple = True)
                    for row, col in zip(index[0],index[1]):
                        label[row, :col] = -100
                    text['labels'] = label
                    return text 
                
                train_dataset = dataset['train'].map( map_function, batched = True, remove_columns = dataset['train'].column_names)
                eval_dataset = dataset['train'].map( map_function, batched = True, remove_columns = dataset['train'].column_names) if 'eval' in list(dataset.keys()) else None 

                return {
                    'train_dataset' : train_dataset, 
                    'eval_dataset' : eval_dataset
                }

            else:
                raise InvalidDatasetFormatError('''
                Dataset does not match expected input-output format
                    dataset = {
                        train : { "question": "...", "context": "...", "answer": "..." },
                        eval : { "question": "...", "context": "...", "answer": "..." }
                    }
                ''')
