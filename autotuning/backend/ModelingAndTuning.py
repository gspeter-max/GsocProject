import sys 
import os 
import numpy as np 

sys.path.append(os.getcwd())
from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        Trainer
        )

import evaluate 
from transformers.trainer_callback import TrainerCallback 
from transformers.training_args import TrainingArguments
from .DatasetUpLoading import UploadDataset
import importlib
from .GlobalConfig import global_config
import subprocess 
import pandas as pd 
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
        self.tokenizer.add_special_tokens({'sep_token': '[SEP]'})
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def ComputeMetrics( self,EvalPredict ):
        
        probability , labels = EvalPredict
        Prediction = probability.argmax(-1)
        PossibleMetrics = ('accuracy', 'f1', 'perplexity')
        labels = np.array(labels, dtype = np.int32)
        Prediction = np.array(Prediction, dtype = np.int32)
        losses = {} 


        mask = labels != -100 
        labels_list = labels[mask]
        pred_list = Prediction[mask]
        
        for metrics in self.HyperparameterConfig.get('ComputeMetricsList',[]):

            if metrics == 'accuracy':

                accuracy = evaluate.load('accuracy')
                accuracy_metrics  = accuracy.compute( references = labels_list.tolist(), predictions = pred_list.tolist())['accuracy']
                losses[metrics] = accuracy_metrics
                

            elif metrics == 'f1':
                f1 = evaluate.load('f1')

                losses[metrics] = f1.compute( references = labels_list.tolist(),
                    predictions = pred_list.tolist(),
                    average = 'macro'
                )['f1']

            elif metrics == 'perplexity':

                from torcheval.metrics.text import Perplexity 
                m = Perplexity() 
                m.update( torch.tensor(probability), torch.tensor(labels)) 
                losses[metrics] = m.compute().items() 
            
            else: 
                raise AttributeError(f'{metrics} not supported , available metrics {PossibleMetrics}')
                
        return losses

    def LoadItTrainIt( self):
        
        Datasets = UploadDataset(
            hf_token = self.HyperparameterConfig['HfToken'], 
            tokenizer = self.tokenizer, 
            FineTuningType = self.HyperparameterConfig['FineTuningType'],
            max_length = self.HyperparameterConfig['ModelSeqMaxLength'], 
            path = self.HyperparameterConfig['DatasetPath']
        )
        
        dataset = Datasets.on_loading()

        quantizationConfig = None
        if (self.HyperparameterConfig.get('QuantizationType4Bit8Bit') == '4bit') or \
            (self.HyperparameterConfig.get('PeftConfig') == 'qlora'):
            quantizationConfig = BitsAndBytesConfig(
                        load_in_4bit = True, 
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type = 'nf4', 
                        bnb_4bit_use_double_quant= True,
                        bnb_4bit_quant_storage= torch.float16
                    )
        
        elif self.HyperparameterConfig.get('QuantizationType4Bit8Bit') == '8bit':
            quantizationConfig = BitsAndBytesConfig(
                load_in_8bit = True
            )

        model = AutoModelForCausalLM.from_pretrained(
                self.HyperparameterConfig.get('ModelName'),
                quantization_config = quantizationConfig,
                device_map = 'auto',
                trust_remote_code = True
                )

        available_layers = set()
        for name, _ in model.named_parameters():
            available_layers.add(str(name).split('.')[-2])
        
        peft_target_modules = self.HyperparameterConfig['PeftConfig']['target_modules']
        
        if not set(peft_target_modules).issubset(available_layers):
            raise ValueError(f'{peft_target_modules} is not found in {available_layers} modules')

        model.resize_token_embeddings(len(self.tokenizer))
        with torch.no_grad():
            model.get_input_embeddings().weight[-1]= torch.mean(model.get_input_embeddings().weight[:-1], dim = 0)

        PeftConfig = self.HyperparameterConfig.get('PeftConfig')
        Lora_config = LoraConfig(
            **PeftConfig
        )
        model = get_peft_model(model, Lora_config)

        training_arg_config = self.HyperparameterConfig.get('TrainingArguments')
        TrainingArg = TrainingArguments(
                **training_arg_config
                )
        
        trainer = Trainer(
                model = model,
                args = TrainingArg,
                train_dataset = dataset['train_dataset'],
                eval_dataset = dataset['train_dataset'],
                compute_metrics = self.ComputeMetrics
            )

        
        if self.HyperparameterConfig.get('EvalSaveFormat')  is not None : 
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
            convertmodel( model, self.tokenizer )

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


