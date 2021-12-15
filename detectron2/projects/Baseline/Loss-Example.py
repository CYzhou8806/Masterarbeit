def mae(disp_gt, disp_est):
	""" Computes the Mean Absolute Error (MAE) also referred to as L1-Norm.

	Difference to predefined mae: Only pixels with GT > 0 are considered.

	@param disp_gt: Tensor containing the reference disparity map(s).
	@param disp_est: Tensor containing the estimated disparity and uncertainty map(s).
	@return: Float value representing the Mean Absolute Error in pixel.
	"""

	diff = K.abs(disp_gt - disp_est)
	diff_nz = tf.boolean_mask(diff, disp_gt)

	if len(diff_nz) == 0:
		return tf.constant(0.0)
	else:
		return K.mean(diff_nz)