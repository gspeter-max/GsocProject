
import torch 
import onnx 
import onnx-tf 


class ConvertModel:
    def __init__( self , TorchToTenserFlow : bool = False , TensorFlowToTorch : bool = False, \
            WhereStored : str = './tfConvetedModel/'): 
        self.TorchToTenserFlow = TorchToTenserFlow 
        self.TensorFlowToTorch = TensorFlowToTorch
        self.WhereStored = WhereStored

    def __call__( self, Model , Input):
        torch.onnx.export(     
                Model , 
                ( Input,), 
                dynamo = True , 
                f = './output',
                verbose = True,
                input_names = ['doc'],
                output_names = ['claim']
            )
        onnxModel = onnx.load('./model_stored')
        PreparedModel = onnx_tf.backend.prepare(onnxModel) 
        PrepareModel.export_graph( self.WhereStored ) 

