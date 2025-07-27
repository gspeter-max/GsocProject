
import torch 
import onnx 
import onnx-tf 


class ConvertModel:
    def __init__( self , TorchToTenserFlow : bool = False , TensorFlowToTorch : bool = False ): 
        self.TorchToTenserFlow = TorchToTenserFlow 
        self.TensorFlowToTorch = TensorFlowToTorch


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
        onnxModel = onnx.load('./output')
        PreparedModel = onnx.backend.prepare(onnxModel) 
        PrepareModel.export(                                                            


         



