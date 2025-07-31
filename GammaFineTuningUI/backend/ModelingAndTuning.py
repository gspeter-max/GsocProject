from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        Trainer
        )


from transformers.trainer_callback import TrainerCallback 
from transformers.training_args import TrainingArguments
from .DatasetUpLoading import UploadDataset
import os 
import importlib
from .GlobalConfig import global_config
import subprocess 
import pandas as pd 

hftoken = os.environ.get('HF_TOKEN')
globalConfig = global_config(
        ModelName = 'gpt2',
        QuantizationType4Bit8Bit = False,
        ComputeMetricsList = ['accuracy_scores','f1_score'],
        HfToken = hftoken, 
        FSDP = True
        )

HyperparameterConfig = globalConfig(
        TokenizationConfig=global_config.GetTokenizationConfig(),
        PeftConfig=global_config.GetPeftConfig(),
        TrainingArguments=global_config.GetTrainingArguments(
            report_to = 'tensorboard',
            fsdp_config = global_config.GetFSDP(), 
            FSDP = globalConfig.FSDP
            )
        ) 

from peft import LoraConfig , get_peft_model, TaskType
import torch
import logging 

logger = logging.getLogger().setLevel(logging.INFO) 

class AllEvaluationResultCallback( TrainerCallback ):
    def __init__(  self, Trainer ):
        super().__init__()
        self.trainer = Trainer 
        self.AllEvaluations = []    


    def on_evaluate( self, args : TrainingArguments, Score, Control ,metrics, **Kwargs ):
        self.AllEvaluations.append( metrics.copy())


class ModelLoadingAndTuning:
    def __init__(self,HyperparameterConfig):
        self.HyperparameterConfig = HyperparameterConfig
        self.tokenizer = AutoTokenizer.from_pretrained( self.HyperparameterConfig['ModelName'])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.add_special_tokens({'additional_special_tokens' : ["[SEP]"]})

    def map_function(self,example):

        text = [f'{doc} [SEP] {claim} ' for doc , claim in zip(example['doc'], example['claim'])]
        tokenized = self.tokenizer(
                text,
                padding = self.HyperparameterConfig.get('TokenizationConfig')['padding'] ,
                return_tensors = 'pt',
                max_length = self.HyperparameterConfig.get('TokenizationConfig')['max_length'],
                truncation = self.HyperparameterConfig.get('TokenizationConfig')['truncation']
        )
        labels = tokenized.input_ids.clone()
        SEPTokenId = self.tokenizer.convert_tokens_to_ids("[SEP]")
        mask_index = torch.argwhere(labels == SEPTokenId )
        rows, columns  = zip(mask_index[0], mask_index[1])
        for r, c  in zip(rows, columns):
                labels[r,:c] = -100
        return {
                'input_ids' : tokenized.input_ids,
                'attention_mask' : tokenized.attention_mask,
                'labels' : labels
                }
    def ComputeMetrics( self,EvalPredict ):
    
        probability , label_ids = EvalPredict
        Prediction = probability.argmax(-1)
        PossibleMetrics = ('accuracy_scores', 'f1_score', 'perplexity')
        losses = {} 
        for metrics in self.HyperparameterConfig.get('ComputeMetricsList'):
            if metrics == 'accuracy_scores':
                from sklearn.metrics import accuracy_score
                
                losses[metrics]  = accuracy_score( label_ids, probability )

            if metrics == 'f1_score':
                from sklearn.metrics import f1_score 

                losses[metrics] = f1_score( label_ids, probability ) 

            if metrics == 'perplexity':
                from torcheval.metrics.text import Perplexity 
                m = Perplexity() 
                m.update( probability, label_ids ) 
                losses[metrics] = m.compute() 
            
            else: 
                raise AttributeError(f'{metrics} not supported , available metrics {PossibleMetrics}')
                
        return losses

    def LoadItTrainIt( self):

        Dataset = UploadDataset(
                ContextOrDocOrPassage  = True, 
                QuestionOrClaimOrUserInput = True, 
                AnswerOrLabelOrResponse = False
        ) 
        dataset = Dataset(self.HyperparameterConfig.get('HfToken'))

        tokenized_data = dataset.map(self.map_function, batched = True, remove_columns = dataset.column_names)
        if self.HyperparameterConfig.get('QuantizationType4Bit8Bit'):
            quantizationConfig = BitsAndBytesConfig(
                        load_in_8bit = True
                    )
        else:
            quantizationConfig = None

        model = AutoModelForCausalLM.from_pretrained(
                self.HyperparameterConfig.get('ModelName'),
                quantization_config = quantizationConfig,
                device_map = 'auto',
                trust_remote_code = True
                )

        model.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            model.get_input_embeddings().weight[-1]= torch.mean(model.get_input_embeddings().weight[:-1], dim = 0)

        PeftConfig = self.HyperparameterConfig.get('PeftConfig')
        Lora_config = LoraConfig(
            **PeftConfig
        )
        model = get_peft_model(model, Lora_config)

        trainingArgConfig = self.HyperparameterConfig.get('TrainingArguments')
        TrainingArg = TrainingArguments(
                **trainingArgConfig
                )
        
        trainer = Trainer(
                model = model,
                args = TrainingArg,
                train_dataset = tokenized_data,
                compute_metrics = self.ComputeMetrics
                )
        
        if self.HyperparameterConfig.get('EvalSaveFormat') not None : 
            if self.HyperparameterConfig.get('EvalSaveFormat').lower() not in ('csv','json'):
                raise AttributeError(f'''{self.HyperparameterConfig.get("EvalSaveFormat")} is supported  ,
                                     acceptable format ("csv","json") '''
                            ) 
            AllEvalResult = AllEvaluationResultCallback( trainer )
            trainer.add_callback( AllEvalResult )

        # %load_ext tensorboard
        # %tensorboard --logdir ./logs
        
        trainer.train()

        if (self.HyperparameterConfig.get('ModelDir') is not None) or (self.HyperparameterConfig.get('SaveFormat') is not None):
            from .GetModel import ConvertModel

            convertmodel = ConvertModel(
                    Format = self.HyperparameterConfig.get('SaveFormat'),
                    WhereStored = self.HyperparameterConfig.get('ModelDir')
                    )
            convertmodel( model, tokenizer )

        pwd = os.getcwd()
        Format = self.HyperparameterConfig.get('EvalSaveFormat') 
        if Format.lower() == 'csv':
            df = pd.DataFrame( AllEvalResult.AllEvaluations )
            df.to_csv('./EvalResult.csv')
            logger.info(f'evaluation result "{pwd}/EvalResult.csv" Stored in {Format.lower()}') 

        if self.HyperparameterConfig.get('EvalSaveFormat').lower() == 'json':
            df = pd.DataFrame( AllEvalResult.AllEvaluations )
            df.to_json('./EvalResult.json')
            logger.info(f'evaluation result  "{pwd}/EvalResult.json" stored in {Format.lower()}') 


# tuning = ModelLoadingAndTuning()
# tuning.LoadItTrainIt()

