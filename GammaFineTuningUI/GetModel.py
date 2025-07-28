
from transformers import TFAutoModelForCausalLM 
import torch 
import onnx 
import onnx-tf 
import shutil

class ConvertModel:
    def __init__( self , GetInTensorFlow : bool = False ,GetInTorch : bool = False, \
            WhereStored : str = './tfConvetedModel/',\
            GetInGGUF : bool = False
                 ): 
        self.GetInTensorflow = GetInTensorflow
        self.GetInTorch = GetInTorch
        self.WhereStored = WhereStored
        self.GetInGGUF = GetInGGUF

    def __call__( self, Model,Tokenizer , Input):
        
        if self.GetInTensorFlow: 
            ptSaveFile = './TempStore', 
            Model.save_pretrained(ptSaveFile) 
            Tokenizer.save_pretrained(ptSaveFile)
    
            model = TFAutoModelForCausalLM.from_pretrained( ptSaveFile, from_pt = True ) 

            model.save_pretrained( self.WhereStored )
            Tokenizer.save_pretrained( self.WhereStored )
            shutil.rmtree(ptSaveFile)

        elif self.GetInTorch: 
            Model.save_pretrained( self.WhereStored )



        return f'model is stored in {self.WhereStored} , {self.GetInTorch} format'

