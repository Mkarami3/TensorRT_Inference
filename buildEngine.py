import engine as eng
import argparse
from onnx import ModelProto 
import tensorrt as trt
from utils.config import config
 
def main(args):
    engine_name = args.plan_file
    onnx_path = args.onnx_file
    batch_size = config.batch_size 
    
    model = ModelProto()
    with open(onnx_path, "rb") as f:
        print("parsing")
        model.ParseFromString(f.read())
 
    d0 = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    d1 = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
    d2 = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value
    d3 = model.graph.input[0].type.tensor_type.shape.dim[4].dim_value
    d4 = model.graph.input[0].type.tensor_type.shape.dim[5].dim_value
    shape = [batch_size , d0, d1 , d2, d3, d4] # for CNN-LSTM model
    #shape = [1, d0, d1 , d2, d3]
    print("data shape required",shape)
    engine = eng.build_engine(onnx_path, shape= shape)
    eng.save_engine(engine, engine_name) 
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_file', type=str, default='models/model_RNN.onnx')
    parser.add_argument('--plan_file', type=str, default='models/inference_engine.plan')
    args=parser.parse_args()
    main(args)
