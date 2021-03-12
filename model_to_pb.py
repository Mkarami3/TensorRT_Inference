import keras
import keras.backend as K
import argparse
K.set_learning_phase(0)
import tensorflow as tf


def keras_to_pb(model, output_filename, output_node_names):
 
    """
    This is the function to convert the keras model to pb.
 
    Args:
       model: The keras model.
       output_filename: The output .pb file name.
       output_node_names: The output nodes of the network (if None, 
       the function gets the last layer name as the output node).
    """
 
    # Get names of input and output nodes.
    in_name = model.layers[0].get_output_at(0).name.split(':')[0]
 
    if output_node_names is None:
        output_node_names = [model.layers[-1].get_output_at(0).name.split(':')[0]]
     
    sess = keras.backend.get_session()
 
    # TensorFlow freeze_graph expects a comma-separated string of output node names.
    output_node_names_tf = ','.join(output_node_names)
 
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)
 
    sess.close()
    wkdir = ''
    tf.train.write_graph(frozen_graph_def, wkdir, output_filename, as_text=False)
 
    return in_name, output_node_names


def main(args):
    # load the model
    model = keras.models.load_model(args.hdf5_file)
    # Freeze the model
    in_tensor_name, out_tensor_names = keras_to_pb(model, args.output_pb_file , None) 
    print("input_layer: ",in_tensor_name)
    print("output_layer: ",out_tensor_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5_file', type=str,default="models/model.h5")
    parser.add_argument('--output_pb_file', type=str, default='models/model_freeze.pb')
    args=parser.parse_args()
    main(args)

