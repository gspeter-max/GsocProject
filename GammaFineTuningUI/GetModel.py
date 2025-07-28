
from transformers import TFAutoModelForCausalLM 
import torch
import logging 
import shutil
import subprocess


logger = logging.getLogger()
logger.setLevel(logging.INFO) 

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
            ptSaveFile = './TempStore'
            Model.save_pretrained(ptSaveFile) 
            Tokenizer.save_pretrained(ptSaveFile)
    
            model = TFAutoModelForCausalLM.from_pretrained( ptSaveFile, from_pt = True ) 

            model.save_pretrained( self.WhereStored )
            Tokenizer.save_pretrained( self.WhereStored )
            shutil.rmtree(ptSaveFile)
            logger.info(''' load out sotred model using huggingface transformer library 
                example :  in tensorflow format loadding 
                    from transformers import TFAutoModelForCausalLM

                    model = TFAutoModelForCausalLM.from_pretrained( WhereStored , from_pt = True )
                '''
                    )

        if self.GetInTorch: 
            Model.save_pretrained( self.WhereStored )

            logger.info(''' load out sotred model using huggingface transformer library 
                    example :  in torch  format loadding 
                        from transformers import AutoModelForCausalLM

                        model = AutoModelForCausalLM.from_pretrained( WhereStored )
                    '''
                    )

        if self.GetInGGUF:
            ptSaveFile = './TempStore'

            model.save_pretrained(ptSaveFile) 
            Tokenizer.save_pretrained( ptSaveFile )

            subprocess.run('git clone https://github.com/ggml-org/llama.cpp.git',shell = True, \
                    stdout = subprocess.PIPE, \
                    stderr = subprocess.PIPE, \
                    text = True 
                ) 
            subprocess.run('pip install -r ./llama.cpp/requirements.txt',\
                    shell = True, \
                    stdout = subprocess.PIPE, \
                    stderr = subprocess.PIPE, \
                    text = True
                )
            subprocess.run(f'mkdir {self.WhereStored}',\
                    shell = True, \
                    stdout = subprocess.PIPE, \
                    stderr = subprocess.PIPE, \
                    text = True
                )

            subprocess.run('python llama.cpp/convert_hf_to_gguf.py ./TempStore --outfile ./GGUF-Model/output_gguf.gguf --outtype q8_0'\
                    shell = True, \
                    stdout = subprocess.PIPE, \
                    stderr = subprocess.PIPE, \
                    text = True 
                )
            logger.info('''
            for loading again 
                do this 
                must be check out version of packages like ( torch , torchvision ) 

                    from transformers import AutoModelForCasualLM
                    import subprocess 

                    pwd = subprocesss.run('pwd', capture_output = True , text = True ).stdout
                    pwd = pwd.strip() 

                    model = AutoModelForCausalLM.from_pretraind( 
                            f"{pwd}/{WhereStored}", \
                            gguf_file = f"{pwd}/{WhereStored}/output_gguf.gguf"
                            )
                '''
                ) 
            
        return f'model is stored in {self.WhereStored} , {self.GetInTorch} format'

