import pyvista as pv
import numpy as np
import json
import matplotlib.pyplot as plt
import time

class Visualize:

	@staticmethod
	def gen_video(disp_pred_tensor, disp_gt_tensor, config):

		file_name = config.sample_file
		#define camera position
		cpos = [(46.99762651516562, 47.59762641979819, 42.997626693979555),
				 (0.0, 0.5999999046325684, -3.9999998211860657),
				 (0.0, 0.0, 1.0)]

		# Define max and min values of disp for the sake of visualization
		disp_mean, disp_std = Visualize.read_json(config.DISP_MEAN)
		d_max = 0.009229241 + 0.18952906
		d_min = 0.009229241 - 0.18952906

		mesh_pv = pv.read(file_name)
		disp = mesh_pv.point_arrays['computedDispl']
		grid = pv.UnstructuredGrid(file_name)
		pts = grid.points 
		indices, unique_z = Visualize.get_indices(file_name) # get the indices of real nodes

		pred_plotter = pv.Plotter()
		pred_plotter.add_mesh(grid, scalars=disp[:, 2], stitle='Z Displacement',show_edges=True, rng=[d_min, d_max],interpolate_before_map=True)
		pred_plotter.add_axes()
		pred_plotter.camera_position = cpos
		pred_plotter.show(interactive=False, auto_close=False, window_size=[800, 600])

		gt_plotter = pv.Plotter()
		gt_plotter.add_mesh(grid, scalars=disp[:, 2], stitle='Z Displacement',show_edges=True, rng=[d_min, d_max],interpolate_before_map=True)
		gt_plotter.add_axes()
		gt_plotter.camera_position = cpos
		gt_plotter.show(interactive=False, auto_close=False, window_size=[800, 600])

		# plotter.open_gif('y_pred.gif')  
		pred_plotter.open_movie('pred.mp4')
		gt_plotter.open_movie('gt.mp4')

		error = []
		disp_pred_sequences = []
		disp_gt_sequences = []

		for i in range(disp_gt_tensor.shape[0]): #loop over batch size

			disp_pred_vect = np.zeros_like(disp)
			disp_gt_vect = np.zeros_like(disp)
			for k in range(disp_pred_vect.shape[0]): # loop over nodes
				index = indices[k]
				disp_pred_vect[k,:] = disp_pred_tensor[i,index[0],index[1], index[2], :]
				disp_gt_vect[k,:] = disp_gt_tensor[i,index[0],index[1], index[2], :]

			disp_pred_vect = (disp_pred_vect*disp_std) + disp_mean	
			pred_plotter.update_coordinates(pts + disp_pred_vect)
			pred_plotter.update_scalars(np.abs(disp_pred_vect[:, 2]))
			pred_plotter.write_frame()

			disp_gt_vect = (disp_gt_vect*disp_std) + disp_mean	
			gt_plotter.update_coordinates(pts + disp_gt_vect)
			gt_plotter.update_scalars(np.abs(disp_gt_vect[:, 2]))
			gt_plotter.write_frame()
			error_snapshot = []
			for z in unique_z:
				disp_pred_vect_filt = disp_pred_vect[((np.round(pts[:,2],2) == z))]
				disp_gt_vect_filt = disp_gt_vect[((np.round(pts[:,2],2) == z))]
				diff_z = np.mean(np.abs(disp_pred_vect_filt - disp_gt_vect_filt),axis=0)
				error_snapshot.append(np.linalg.norm(diff_z))

			disp_gt_sequences.append(disp_gt_vect)
			disp_pred_sequences.append(disp_pred_vect)
			error.append(error_snapshot)

		pred_plotter.close()
		gt_plotter.close()

		disp_gt_mean = np.linalg.norm(np.mean(disp_gt_sequences, axis=0), axis=1)
		disp_pred_mean = np.linalg.norm(np.mean(disp_pred_sequences, axis=0), axis=1)
		print(disp_gt_mean.shape)
		error_np = np.array([np.array(i) for i in error])
		print("[INFO] average error at the different heights...")
		print(np.mean(error_np, axis=0))

		Visualize.bland_altman_plot(disp_gt_mean,disp_pred_mean)
		

	@staticmethod
	def read_json(json_file):

		dic = json.loads(open(json_file).read())

		mean_x = float(list(dic.values())[0])
		mean_y = float(list(dic.values())[1])
		mean_z = float(list(dic.values())[2])
		std_x = float(list(dic.values())[3])
		std_y = float(list(dic.values())[4])
		std_z = float(list(dic.values())[5])       
		mean_array = np.array([mean_x, mean_y, mean_z])
		std_array = np.array([std_x, std_y, std_z])

		return mean_array, std_array

	@staticmethod
	def get_indices(file_name):

		mesh_pv = pv.read(file_name)

		pts = mesh_pv.points
		pts_x = np.unique(pts[:,0])
		pts_y = np.unique(pts[:,1])
		pts_z = np.unique(pts[:,2])
		indices = []

		for i in range(pts.shape[0]):
			x = pts[i, 0]
			y = pts[i, 1]
			z = pts[i, 2]

			x_index = np.where(pts_x == x)[0][0]
			y_index = np.where(pts_y == y)[0][0]
			z_index = np.where(pts_z == z)[0][0]

			indices.append([x_index, y_index, z_index])

		return indices,pts_z

	@staticmethod
	def bland_altman_plot(data_gt, data_pred):
		
		data_gt     = np.asarray(data_gt)
		data_pred     = np.asarray(data_pred)
		mean      = np.mean([data_gt, data_pred], axis=0)
		diff      = data_gt - data_pred                   # Difference between data1 and data2
		md        = np.mean(diff)                   # Mean of the difference
		sd        = np.std(diff, axis=0)            # Standard deviation of the difference

		plt.scatter(data_gt, diff)
		plt.axhline(md,           color='gray', linestyle='--')
		plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
		plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
		np.savetxt('mean.out', mean, delimiter=',')
		np.savetxt('diff.out', diff, delimiter=',')
		plt.savefig('Figure1.png')
		plt.show()


