from utils.config import config
from utils.io import HDF5DatasetGenerator
from utils.visualize import Visualize
import numpy as np
import tensorrt as trt
import engine as eng
import inference as inf
import argparse
import time

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.batch_size, config.FORCE_MEAN, config.DISP_MEAN) #batch_size
gen = valGen.generator()

def main(args):

    serialized_plan_fp32 = args.engine_file
    print("[INFO] Loading Engine...")
    engine = eng.load_engine(trt_runtime, serialized_plan_fp32)
    print("[INFO] Allocate Buffer...")
    

    print("[INFO] Apply Inference...")
    disp_tensors_pred = []
    disp_tensors_gt = []
    for i in range(config.NUM_VAL//config.batch_size):
        (force_tensor,disp_tensor_gt) = next(gen)
        h_input, d_input, h_output, d_output, stream = inf.allocate_buffers(engine, config.batch_size, trt.float16)#batch_size
        start = time.time()
        TensorRT_pred = inf.do_inference(engine, force_tensor, h_input, d_input, h_output, d_output, stream, config.batch_size) #batch_size
        end = time.time()
        print("inference time including buffer copy", end-start)
        print("TensorRT_pred",TensorRT_pred.shape)
        disp_tensors_pred.append(TensorRT_pred)
        disp_tensors_gt.append(disp_tensor_gt)
        #break

    disp_tensors_pred = np.asarray(disp_tensors_pred).reshape(-1,config.data_shape[0],config.data_shape[1],config.data_shape[2],config.data_shape[3])
    disp_tensors_gt = np.asarray(disp_tensors_gt).reshape(-1,config.data_shape[0],config.data_shape[1],config.data_shape[2],config.data_shape[3])
    print(disp_tensors_pred.shape)
    Visualize.gen_video( disp_tensors_pred, disp_tensors_gt, config) #visualize the results
    
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine_file', type=str,default="/home/mohammad/NRC_David/TensorRT_Inference/models/inference_engine.plan")
    args=parser.parse_args()
    main(args)
