from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        BitsAndBytesConfig, 
        TrainingArguments, 
        Trainer
        )
from GlobalConfig import GetIt
globalConfig = GetIt(
        ModelName = 'google/gemma-3-1b-pt',
        QuantizationType4Bit8Bit = False 
        )

tokenizerConfig = globalConfig.GetTokenizationConfig()
peftConfig = globalConfig.GetPeftConfig() 
TrainingArgument = globalConfig.GetTrainingArguments() 
HyperparameterConfig = globalConfig(
        tokenizerConfig, 
        peftConfig, 
        TrainingArgument
        )


from peft import LoraConfig , get_peft_model, TaskType 
import torch 

tokenizer = AutoTokenizer.from_pretrained(HyperparameterConfig['ModelName'])
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'additional_special_tokens' : ["[SEP]"]}) 

def map_function(example):

    text = [f'{doc} [SEP] {claim} ' for doc , claim in zip(example['doc'], example['claim'])]
    tokenized = tokenizer( 
                    text, 
                    padding = HyperparameterConfig.get('TokenizationConfig')['padding'] ,
                    return_tensors = 'pt', 
                    max_length = HyperparameterConfig.get('TokenizationConfig')['max_length'], 
                    truncation = HyperparameterConfig.get('TokenizationConfig')['truncation']
                )

    labels = tokenized.input_ids.clone()
    SEPTokenId = tokenizer.convert_token_to_ids("[SEP]")
    mask_index = torch.argwhere(labels == SEPTokenId )
    rows, columns  = zip(mask_index[0], mask_index[1])
    for r, c  in zip(rows, columns):
        labels[r,:c] = -100
    return {
            'input_ids' : tokenized.input_ids, 
            'attention_mask' : tokenized.attention_mask, 
            'labels' : labels
            }

tokenized_data = dataset.map(map_function, batched = True, remove_columns = dataset.column_names)
class ModelLoadingAndTuning:
    def __init__(self):
        pass

    def LoadItTrainIt( self, dataset ):

        if HyperparameterConfig.get('QuantizationType4Bit8Bit'):
            quantizationConfig = BitsAndBytesConfig(
                        load_in_8bit = True
                    )
        else: 
            quantizationConfig = None
        
        model = AutoModelForCausalLM.from_pretrained(
                HyperparameterConfig.get('ModelName'),
                quantization_config = quantizationConfig,
                device_map = 'auto',
                trust_remote_code = True
                )

        model.resize_token_embeddings(len(tokenizer))
        with no_grad(): 
            model.get_input_embeddings().weight[-1]= torch.mean(model.get_input_embeddings().weight[:-1], dim = 0)

        PeftConfig = HyperparameterConfig.get('PeftConfig')
        Lora_config = LoraConfig(
            **PeftConfig
        )
        model = get_peft_model(model, Lora_config)

        trainingArgConfig = HyperparameterConfig.get('TrainingArguments')
        TrainingArg = TrainingArguments(
                **trainingArgConfig
                )

        trainer = Trainer(
                model = model,
                args = TrainingArg
                train_dataset = tokenized_data 
                )
        trainer.train()
