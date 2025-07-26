from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        BitsAndBytesConfig, 
        TrainingArguments, 
        Trainer
        ) 

from peft import LoraConfig , get_peft_model, TaskType 
import torch 

tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-1b-pt')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'additional_special_tokens' : ["[SEP]"]}) 

def map_function(example):

    text = [f'{doc} [SEP] {claim} ' for doc , claim in zip(example['doc'], example['claim'])]
    tokenized = tokenizer( text, padding = 'max_length' ,return_tensors = 'pt' , max_length = 128, truncation = True )

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
        quantizationConfig = BitsAndBytesConfig(
                    load_in_8bit = True
                )
        model = AutoModelForCausalLM.from_pretrained(
                'google/gemma-3-1b-pt',
                quantization_config = quantizationConfig,
                device_map = 'auto',
                trust_remote_code = True
                ) 
        model.resize_token_embeddings(len(tokenizer))
        with no_grad(): 
            model.get_input_embeddings().weight[-1]= torch.mean(model.get_input_embeddings().weight[:-1], dim = 0)  

        Lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            task_type = TaskType.CAUSAL_LM,
            target_modules = ["q_proj", "v_proj"]
        )
        model = get_peft_model(model, Lora_config)
        TrainingArg = TrainingArguments(
                output_dir = './output',
                do_train = True, 
                label_names = ['claim_input_ids']
                )

        trainer = Trainer(
                model = model,
                args = TrainingArg,
                train_dataset = tokenized_data 
                )
        trainer.train()
