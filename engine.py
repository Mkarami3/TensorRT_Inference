from utils.config import config
import tensorrt as trt
import os

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)
def build_engine(onnx_path, shape):

   """
   This is the function to create the TensorRT engine
   Args:
      onnx_path : Path to onnx_file. 
      shape : Shape of the input of the ONNX file. 
  """
   with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:


       # Parse model file
       if not os.path.exists(onnx_path):
            print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_path))
            exit(0)
       print('Loading ONNX file from path {}...'.format(onnx_path))
       with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
            print('Beginning ONNX file parsing')
            parser.parse(model.read())
       print('Completed parsing of ONNX file')

       builder.max_workspace_size = (256 << 20)
       builder.max_batch_size = config.batch_size
       builder.fp16_mode = False
       with open(onnx_path, 'rb') as model:
           parser.parse(model.read())
       network.get_input(0).shape = shape
       engine = builder.build_cuda_engine(network)
       return engine

def save_engine(engine, file_name):
   buf = engine.serialize()
   with open(file_name, 'wb') as f:
       f.write(buf)

def load_engine(trt_runtime, plan_path):
   with open(plan_path, 'rb') as f:
       engine_data = f.read()
   engine = trt_runtime.deserialize_cuda_engine(engine_data)
   return engine

