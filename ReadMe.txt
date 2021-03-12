Steps to convert Tensorflow model to TensorRT model and running inference

1. Freeze the model---> KerasModel_to_pb.py

2. Convert the frozen model to onnx format, run the following command in terminal 

For CNN model
python3 -m tf2onnx.convert  --input /home/mohammad/NRC_David/TensorRT_Inference/models/model_freeze.pb --inputs input_1:0 --outputs conv3d_9/add:0 --output model_RNN.onnx

For CNN-LSTM model
python3 -m tf2onnx.convert  --input /home/mohammad/NRC_David/TensorRT_Inference/models/model_freeze.pb --inputs input_1:0 --outputs time_distributed_7/Reshape_2:0 --output model_RNN.onnx

3. Create the Tensorrt inference engine--> buildEngine.py

4. run the inference engine--> run_inference.py


****To change the batch_size, go to utils/config/config.py and change the batch_size value*****
