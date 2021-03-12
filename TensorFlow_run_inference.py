from utils.config import config
from utils.io import HDF5DatasetGenerator
from utils.visualize import Visualize
import numpy as np
import keras
import keras.backend as K
import argparse
import tensorflow as tf    
import argparse
import time

from tensorflow.compat.v1.keras.backend import set_session
config1 = tf.compat.v1.ConfigProto()
config1.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config1.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config1)
set_session(sess)
K.set_learning_phase(0)

valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.batch_size, config.FORCE_MEAN, config.DISP_MEAN) #batch_size
gen = valGen.generator()

def load_frozen_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def main(args):  

    disp_tensors_pred = []
    disp_tensors_gt = []
    model = keras.models.load_model(args.keras_model)
    for i in range(config.NUM_VAL//config.batch_size):
        (force_tensor,disp_tensor_gt) = next(gen)
        print("batch size:", force_tensor.shape[0])
        #print("[INFO] keras predict...")
        start = time.time()
        Keras_pred = model.predict(force_tensor)
        disp_tensors_pred.append(Keras_pred)
        disp_tensors_gt.append(disp_tensor_gt)
        end = time.time()
        break

    disp_tensors_pred = np.asarray(disp_tensors_pred).reshape(-1,config.data_shape[0],config.data_shape[1],config.data_shape[2],config.data_shape[3])
    disp_tensors_gt = np.asarray(disp_tensors_gt).reshape(-1,config.data_shape[0],config.data_shape[1],config.data_shape[2],config.data_shape[3])
    #Visualize.gen_video(disp_tensors_pred, disp_tensors_gt, config) #visualize the results
    #print("disp shape",disp_tensors_gt.shape)

    print("[INFO] forzen keras predict...")
    graph = load_frozen_graph(config.FREEZE_MODEL_PATH)
    x = graph.get_tensor_by_name('prefix/input_1_1:0')
    y = graph.get_tensor_by_name('prefix/conv3d_9_1/add:0')
    with tf.Session(graph=graph) as sess:
        # Note: we don't nee to initialize/restore anything
        # There is no Variables in this graph, only hardcoded constants 
        start1 = time.time()
        y_out = sess.run(y, feed_dict={x: force_tensor})
        end1 = time.time()
        print("inference time by fronzen model for one batch:", (end1 - start1))

    print("Inference time by keras model for one batch:", (end - start))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--keras_model', type=str,default="/home/mohammad/NRC_David/TensorRT_Inference/models/model_UNet.h5")
    args=parser.parse_args()
    main(args)
