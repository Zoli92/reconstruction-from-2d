__kernel void calculateDissim(
	__global float* left_image,
	__global float* right_image,
	__global float* dissim_matrix,
	int image_width,
	int image_height,
	int patch_size,
	int current_row
	)
{	
	int left = get_global_id(0);
	int right = get_global_id(1);

	
	dissim_matrix[left * image_width + right] = fabs(left_image[(current_row) * image_width + left] - right_image[(current_row ) * image_width + right]);

	//for (int i = -patch_size; i <= path_size; i++)
	//{
	//	for (int j = -patch_size; j <= patch_size; j++)
	//	{
	//		if (current_row +i >= 0 && current_row+i < image_width && right +j >= 0 && right+j < image_width && left + j >= 0 && left + j < 319)
	//		{
	//			dissim_matrix[left * image_width + right] += fabs(left_image[(current_row + i) * image_width + left + j] - right_image[(current_row + i) * image_width + j + right])* fabs(left_image[(current_row + i) * image_width + left + j] - right_image[(current_row + i) * image_width + j + right]);
	//		}
	//	}
	//}

}
__kernel void calculateC(
	__global float* left_image,
	__global float* right_image,
	__global float* dissim_matrix,
	const int wave,
	__global float* cost_matrix,
	__global int* node_matrix,
	const int wave_length,
	const int over_half
)
{

	float lambda =400;
	int id = get_global_id(0);
	int idx = 0;
	if (over_half == 0)
	{
		idx = id * 450 + wave - id;
		//idx = id * 400 + wave - id;
	}
	else {
		idx = 450 * ((wave % 450) + id) + 449 - id;
		//idx = 400 * ((wave % 400) + id) + 399 - id;
	}
	
	
	int left_idx = idx - 1;

	int upper_idx = idx - 400;
	/*int upper_idx = idx - 400;
	int upper_left_idx = idx - 401;*/
	int upper_left_idx = idx - 401;
	if (wave == 0)
	{
		cost_matrix[0] = dissim_matrix[0];
	}
	else if (left_idx / 400 != idx/400)
	//else if (left_idx / 400 != idx/400)
	{
		//printf("ONLY UP %d, %d choosing: %d\n", idx, wave, upper_idx);
		cost_matrix[idx] = cost_matrix[upper_idx] + lambda;//C[wave - 2][id - 1],C[wave - 1][id - 1] + lambda, C[wave - 1][id]
		
	}
	else if (upper_idx < 0)
	{
		//printf("ONLY LEFT %d, %d choosing :%d\n", idx, wave, left_idx);
		cost_matrix[idx] = cost_matrix[left_idx] + lambda;//C[wave - 2][id - 1],C[wave - 1][id - 1] + lambda, C[wave - 1][id]
	}
	else
	{
		cost_matrix[idx] = min(min(cost_matrix[upper_left_idx] + dissim_matrix[idx], cost_matrix[upper_idx] + lambda), cost_matrix[left_idx] + lambda);//C[wave - 2][id - 1],C[wave - 1][id - 1] + lambda, C[wave - 1][id]
	
	}
	
	if(cost_matrix[idx] == cost_matrix[upper_left_idx] + dissim_matrix[idx])
	{
		node_matrix[idx] = 0;
	}
	else if (cost_matrix[idx] == cost_matrix[upper_idx] + lambda)
	{
		node_matrix[idx] = 1; //LEFT
	}
	else if(cost_matrix[idx] == cost_matrix[left_idx] + lambda)
	{
		node_matrix[idx] = 2; //RIGHT
	}
}

__kernel void templateMatching(
	__global float* left_image,
	__global float* right_image,
	__global float* disparity_map,
	const int width,
	const int height,
	const int patch_size
	)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	float min_cost = FLT_MAX;
	int best_disparity = 0;

	for (int d = 100; d < 300; ++d) 
	{
		float cost = 0.0f;

		for (int wy = -patch_size; wy <= patch_size; wy++)
		{
			for(int wx = -patch_size; wx <= patch_size; wx++)
			{
				int lx = x + wx;
				int ly = y + wy;
				int rx = x - d + wx;
				int ry = y + wy;
				//if (lx >= 0 && ly >= 0 && lx < width && ly < height && rx >= 0 && ry >= 0 && rx < width && ry < height)
				{
					int idx_l = ly * width + lx;
					int idx_r = ry * width + rx;
					float diff = left_image[idx_l] - right_image[idx_r];
					cost += diff * diff;
				}
			}
		}
		if (cost < min_cost) 
		{
			min_cost = cost;
			best_disparity = d;
		}
	}
	
	disparity_map[y * width + x] = (float)best_disparity;
}

__kernel void pointMSE(
	__global float* original_image,
	__global float* created_image,
	__global float* result,
	const int width,
	const int height
)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	result[y * width + x] = (original_image[y * width + x] - created_image[y * width + x]) * (original_image[y * width + x] - created_image[y * width + x])/(2* height * width);
	
}

__kernel void Reduce(
	__global float* input,
	__global float* result,
	__local float* scratch
)
{
	int g_id = get_global_id(0);
	int l_id = get_local_id(0);
	scratch[l_id] = input[g_id];
	result[g_id] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int offset = get_local_size(0) / 2; offset != 0; offset /= 2)
	{
		if (l_id < offset)
		{
			__local float* left = scratch + l_id;
			__local float* right = left + offset;
			*left += *right;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (l_id == 0) {
		result[get_group_id(0)] = scratch[0];
	}
	
}
__kernel void pointCloud(
	__global float* disparity,
	__global float* result,
	const int focal_length,
	const int baseline,
	const int image_width
)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	int idx = y * image_width + x;

	float d = disparity[idx];
	if (d <= 0) {
		result[idx * 4 + 0] = 0.0f; 
		result[idx * 4 + 1] = 0.0f; 
		result[idx * 4 + 2] = 0.0f; 
		result[idx * 4 + 3] = 0.0f;
		return;
	}
	float Z = focal_length * baseline / d;
	float X = -baseline * (2 * x + d) / (2 * d);
	float Y = baseline * y / d;

	result[idx * 4 + 0] = X;
	result[idx * 4 + 1] = Y;
	result[idx * 4 + 2] = Z;
	result[idx * 4 + 3] = d;

}
