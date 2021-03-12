import pyvista as pv
import numpy as np

class Preprocessor:

	@staticmethod
	def array_reshape(meshfile, data_shape,channel_firtst=True):

		mesh_pv = pv.read(meshfile)

		disps = mesh_pv.point_arrays['computedDispl']
		forces = mesh_pv.point_arrays['externalForce']
		pts = mesh_pv.points

		pts_x = np.unique(pts[:,0])
		pts_y = np.unique(pts[:,1])
		pts_z = np.unique(pts[:,2])

		forces_reshape = np.zeros(data_shape)
		disps_reshape = np.zeros(data_shape)

		for i in range(forces.shape[0]):
			x = pts[i, 0]
			y = pts[i, 1]
			z = pts[i, 2]

			x_index = np.where(pts_x == x)[0][0]
			y_index = np.where(pts_y == y)[0][0]
			z_index = np.where(pts_z == z)[0][0]

			if channel_firtst:
				forces_reshape[:, x_index, y_index, z_index] = forces[i, :]
				disps_reshape[:, x_index, y_index, z_index] = disps[i, :]

			else:
				forces_reshape[x_index, y_index, z_index, :] = forces[i, :]
				disps_reshape[x_index, y_index, z_index, :] = disps[i, :]

		force_mean= np.mean(np.abs(forces), axis=0)
		disp_mean = np.mean(np.abs(disps), axis=0)
		force_std = np.std(np.abs(forces), axis=0)
		disp_std = np.std(np.abs(disps), axis=0)

		return forces_reshape, disps_reshape, force_mean, disp_mean, force_std, disp_std

