from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        Trainer
        )


from transformers.trainer_callback import TrainerCallback , TrainingArguments 
from .DatasetUpLoading import UploadDataset
import os 
import importlib
from .GlobalConfig import GetIt
import subprocess 
import pandas as pd 

hftoken = os.environ.get('HF_TOKEN')
globalConfig = GetIt(
        ModelName = 'gpt2',
        QuantizationType4Bit8Bit = False,
        ComputeMetricsList = ['accuracy_scores','f1_score'],
        HfToken = hftoken, 
        FSDP = True
        )

HyperparameterConfig = globalConfig(
        TokenizationConfig=GetIt.GetTokenizationConfig(),
        PeftConfig=GetIt.GetPeftConfig(),
        TrainingArguments=GetIt.GetTrainingArguments(
            report_to = 'tensorboard',
            fsdp_config = GetIt.GetFSDP(), 
            FSDP = globalConfig.FSDP
            )
        ) 

from peft import LoraConfig , get_peft_model, TaskType
import torch

tokenizer = AutoTokenizer.from_pretrained(HyperparameterConfig['ModelName'])
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'additional_special_tokens' : ["[SEP]"]})

class ModelLoadingAndTuning:
    def __init__(self,HyperparameterConfig):
        self.HyperparameterConfig = HyperparameterConfig

    def map_function(self,example):

            text = [f'{doc} [SEP] {claim} ' for doc , claim in zip(example['doc'], example['claim'])]
            tokenized = tokenizer(
                    text,
                    padding = self.HyperparameterConfig.get('TokenizationConfig')['padding'] ,
                    return_tensors = 'pt',
                    max_length = self.HyperparameterConfig.get('TokenizationConfig')['max_length'],
                    truncation = self.HyperparameterConfig.get('TokenizationConfig')['truncation']
            )
            labels = tokenized.input_ids.clone()
            SEPTokenId = tokenizer.convert_tokens_to_ids("[SEP]")
            mask_index = torch.argwhere(labels == SEPTokenId )
            rows, columns  = zip(mask_index[0], mask_index[1])
            for r, c  in zip(rows, columns):
                    labels[r,:c] = -100
            return {
                    'input_ids' : tokenized.input_ids,
                    'attention_mask' : tokenized.attention_mask,
                    'labels' : labels
                    }
    
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
        
        def ComputeMetrics( self,EvalPredict ):
    
            probability , label_ids = EvalPredict
            Prediction = probability.argmax(-1)
            PossibleMetrics = ('accuracy_scores', 'f1_score', 'perplexity')
            losses = {} 
            for metrics in self.HyperparameterConfig.get('ComputeMetircsList'):
                if metrics == 'accuracy_scroe':
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

        trainingArgConfig = self.HyperparameterConfig.get('TrainingArguments')
        TrainingArg = TrainingArguments(
                **trainingArgConfig
                )
        
        class AllEvaluationResultCallback( TrainerCallback ):
            def __init__(  self, Trainer ):
                super().__init__()
                self.trainer = Trainer 
                self.AllEvaluations = [] 

            def __call__( self, args : TrainingArguments, Score, Control , **Kwargs ):
                LossResult = self.trainer.evaluate( self.trainer.eval_datasets ) 
                self.AllEvaluations.append( LossResult )
        
        trainer = Trainer(
                model = model,
                args = TrainingArg,
                train_dataset = tokenized_data,
                compute_metrics = self.ComputeMetrics
                )
        
        if self.Hyperparameter.get('EvalSaveFormat') not None : 
            if self.Hyperparameter.get('EvalSaveFormat').lower() not in ('csv','json'):
                raise AttributeError(f'''{self.Hyperparameter.get("EvalSaveFormat")} is supported  ,
                                     acceptable format ("csv","json") '''
                            ) 
            AllEvalResult = AllEvaluationResultCallback( trainer )
            trainer.add_callback( AllEvalResult )

        # %load_ext tensorboard
        # %tensorboard --logdir ./logs
        
        trainer.train()

        if (self.HyperparameterConfig.get('ModelDir') is not None) or (self.HyperparameterConfig.get('SaveFormat') is not None):
            from backend.GetModel import ConvertModel

            convertmodel = ConvertModel(
                    Format = self.HyperparameterConfig.get('SaveFormat'),
                    WhereStored = self.HyperparameterConfig.get('ModelDir')
                    )
            convertmodel()

        pwd = subprocess.run( 'pwd', shell = True , text = True, capture_output = True ).stdout
        Format = self.Hyperparameter.get('EvalSaveFormat') 
        if Format.lower() == 'csv':
            df = pd.DataFrame( AllEvalResult.AllEvaluations )
            df.to_csv('./EvalResult.csv')
            logger.info(f'evaluation result "{pwd}/EvalResult.csv" Stored in {Format.lower()}') 

        if self.Hyperparameter.get('EvalSaveFormat').lower() == 'json':
            df = pd.DataFrame( AllEvalResult.AllEvaluations )
            df.to_json('./EvalResult.json')
            logger.info(f'evaluation result  "{pwd}/EvalResult.json" stored in {Format.lower()}') 


# tuning = ModelLoadingAndTuning()
# tuning.LoadItTrainIt()
