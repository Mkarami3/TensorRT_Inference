import keras
from keras2onnx import convert_keras
import argparse

#free model to onnx format run the following command in terminal
#python3 -m tf2onnx.convert  --input /home/mohammad/NRC_David/UNet_FEM/models/model_freeze.pb --inputs input_1:0 --outputs conv3d_9/add:0 --output model.onnx
def keras_to_onnx(model, output_filename):
   onnx = convert_keras(model, output_filename)
   with open(output_filename, "wb") as f:
       f.write(onnx.SerializeToString())


def main(args):
    model = keras.models.load_model(args.freeze_file)
    keras_to_onnx(model, args.onnx_file) 
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze_file', type=str,default="models/model_RNN.h5")
    parser.add_argument('--onnx_file', type=str, default='models/model.onnx')
    args=parser.parse_args()
    main(args)
