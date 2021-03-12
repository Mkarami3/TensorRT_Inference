NUM_VAL = 300
NUM_TEST = 1
batch_size = 1

time_step = 2 # time step size is required only for RNN_FEM model

data_shape = (18, 18, 8, 3) #input data shape required for UNet model
data_shape_RNN = (time_step, 18, 18, 8, 3) #input data shape required for RNN model

# dataset path for reading vtk files
data_path = "/home/mohammad/NRC_David/dataset/"
sample_file = "/home/mohammad/NRC_David/dataset/fem_coarse_mesh_gpu_0011200.vtk"
# define the path to the output directory used for storing plots,
TRAIN_HDF5 = "/home/mohammad/NRC_David/UNet_FEM/utils/HDF5_files/train.hdf5"
VAL_HDF5 = "/home/mohammad/NRC_David/UNet_FEM/utils/HDF5_files/val.hdf5"
TEST_HDF5 = "/home/mohammad/NRC_David/UNet_FEM/utils/HDF5_files/test.hdf5"
MODEL_PATH = "/home/mohammad/NRC_David/UNet_FEM/utils/HDF5_files/model.h5"
OUTPUT_PATH = "/home/mohammad/NRC_David/UNet_FEM/utils/HDF5_files/"
FREEZE_MODEL_PATH = "/home/mohammad/NRC_David/UNet_FEM/utils/HDF5_files/model_freeze.pb"

FORCE_MEAN = "/home/mohammad/NRC_David/UNet_FEM/utils/HDF5_files/FORCE_MEAN.json"
DISP_MEAN = "/home/mohammad/NRC_David/UNet_FEM/utils/HDF5_files/DISP_MEAN.json"
