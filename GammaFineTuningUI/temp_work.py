from transformers import AutoTokenizer
import torch
import copy


tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-1b-pt')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'additional_special_tokens' : ['[SEP]']})

def map_function(example):


    text = [f'{doc} [SEP] {claim}' for doc , claim in zip(example['doc'], example['claim'])]
    tokenized = tokenizer( text, padding = 'max_length' ,return_tensors = 'pt', max_length = 128, truncation = True )

    labels = tokenized.input_ids.clone()
    SEPTokenId = tokenizer.convert_tokens_to_ids('[SEP]')

    bool_mask = labels == SEPTokenId
    mask_index = torch.argwhere(bool_mask)

    rows, columns  = zip(mask_index[0], mask_index[1])

    for r, c  in zip(rows, columns):
        labels[r,:c] = -100
    
    return {
            'input_ids' : tokenized.input_ids,
            'attention_mask' : tokenized.attention_mask,
            'labels' : labels
            }

tokenized_data = dataset.map(map_function, batched = True, remove_columns = dataset.column_names)
tokenized_data.set_format( 
                          device = 'cuda'
                          ) 

