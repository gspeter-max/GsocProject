
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
        ''' 
        import tensorflow as tf

        #Load the model from thedirectory
        loaded_tf_model = tf.saved_model.load('/content/main')

        # You can now use the loaded model to make predictions
        # For example, if you have some input data:
        # input_data = ...
        # predictions = loaded_tf_model(input_data)
        import numpy as np

        # Create some dummy input data with the correct shape
        input_data = np.random.rand(1, 10).astype(np.float32)

        # Get the output from the model
        predictions = loaded_tf_model(x=input_data)

        # Print the predictions
        print(predictions)
        ''' 

