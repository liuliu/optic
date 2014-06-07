#include <cuda.h>
#include <assert.h>
#include <stdio.h>

template <int channel_per_thread, int filter_per_thread, int batch_per_block>
__global__ static void _cwc_kern_convolutional_backward_propagate_coefficient_default(const int strides, const int border, const int batch,
		float* input, const int rows, const int cols, const int channels,
		float* out_grad, const int out_rows, const int out_cols,
		float* coeff, const int filter_rows, const int filter_cols, const int count)
{
	assert(gridDim.x == filter_cols);
	assert(gridDim.y == filter_rows);
	extern __shared__ float shared[];
	float* shared_input = &shared[0];
	float* shared_out_grad = &shared[channels * batch_per_block];
	const int thidx = threadIdx.x + threadIdx.y * blockDim.x;
	const int thcnt = blockDim.x * blockDim.y;
	assert(blockDim.x * filter_per_thread == count);
	assert(blockDim.y * channel_per_thread == channels);
	assert(thcnt >= channels * batch_per_block);
	assert(thcnt >= count);
	const int origin_x = blockIdx.x;
	const int origin_y = blockIdx.y;
	const int batch_group_idx = blockIdx.z / out_rows;
	const int start_x = max(origin_x - border, 0) - (origin_x - border);
	const int end_x = min(out_cols, (cols + border - origin_x + strides - 1) / strides);
	input += (rows * cols * channels * batch_group_idx + (origin_y * cols + origin_x) * channels) * batch_per_block;
	out_grad += out_rows * out_cols * count * batch_group_idx * batch_per_block;
	int i, j, c, x;
	const int y = blockIdx.z % out_rows;
	float prod[channel_per_thread][filter_per_thread];
	#pragma unroll
	for (i = 0; i < channel_per_thread; i++)
		#pragma unroll
		for (j = 0; j < filter_per_thread; j++)
			prod[i][j] = 0;
	const int iy = origin_y + y * strides - border;
	if (iy >= 0 && iy < rows)
	{
		input += (y * strides - border) * cols * channels * batch_per_block;
		out_grad += y * out_cols * count * batch_per_block;
		for (x = start_x; x < end_x; x++)
		{
			if (thidx < count)
				#pragma unroll
				for (c = 0; c < batch_per_block; c++)
					shared_out_grad[c * count + thidx] = out_grad[x * count * batch_per_block + c * count + thidx];
			if (thidx < channels * batch_per_block)
				shared_input[thidx] = input[(x * strides - border) * channels * batch_per_block + thidx];
			__syncthreads();
			#pragma unroll
			for (i = 0; i < channel_per_thread; i++)
				#pragma unroll
				for (j = 0; j < filter_per_thread; j++)
				{
					float sum = 0;
					#pragma unroll
					for (c = 0; c < batch_per_block; c++)
						sum += shared_input[c * channels + i + threadIdx.y * channel_per_thread] * shared_out_grad[c * count + j + threadIdx.x * filter_per_thread];
					prod[i][j] += sum;
				}
			__syncthreads();
		}
	}
	const int cocnt = filter_cols * filter_rows * count;
	coeff += cocnt * channels * blockIdx.z + (origin_y * filter_cols + origin_x) * count;
	#pragma unroll
	for (i = 0; i < channel_per_thread; i++)
		#pragma unroll
		for (j = 0; j < filter_per_thread; j++)
			coeff[(i + threadIdx.y * channel_per_thread) * cocnt + j + threadIdx.x * filter_per_thread] = prod[i][j];
}

/*
__global__ static void _cwc_kern_convolutional_backward_propagate_coefficient_whole(const int strides, const int border, const int batch,
		float* input, const int rows, const int cols, const int channels,
		float* out_grad, const int out_rows, const int out_cols,
		float* coeff, const int filter_rows, const int filter_cols, const int count)
{
	assert(gridDim.x == channels);
	assert(gridDim.y == count);
	extern __shared__ float shared[];
	float* shared_input = &shared[0];
	float* shared_out_grad = &shared[(cols + border * 2) * filter_rows * 4];
	const int thidx = threadIdx.x + threadIdx.y * blockDim.x;
	const int batch_group_idx = blockIdx.z;
	const int origin_x = threadIdx.x;
	const int origin_y = threadIdx.y;
	const int channel_idx = blockIdx.x;
	const int count_idx = blockIdx.y;
	input += channel_idx * rows * cols * batch_group_idx * 4;
	out_grad += count_idx * out_rows * out_cols * batch_group_idx * 4;
	int i, j, c;
	float prod = 0;
	for (i = 0; i < filter_rows - 1 - border; i++)
		for (j = 0; j < cols; j += 8)
			if (thidx < 32)
				shared_input[(i * (cols + border * 2) + j + border) * 4 + thidx] = input[(i * cols + j) * 4 + thidx];
	for (i = 0; i < out_rows; i++)
	{
		if (thidx < 32)
			for (j = 0; j < out_cols; j += 8)
				shared_out_grad[j * 4 + thidx] = out_grad[j * 4 + thidx];
		if (thidx < 32)
			#pragma unroll
			for (c = 0; c < strides; c++)
				#pragma unroll
				for (j = 0; j < cols; j += 8)
					shared_input[(((i * strides + c + filter_rows - 1 - border) % filter_rows) * (cols + border * 2) + j + border) * 4 + thidx] = input[(c * cols + j) * 4 + thidx];
		__syncthreads();
		float* input_thread = shared_input + ((origin_y + i * strides) % filter_rows) * (cols + border * 2) * 4 + origin_x;
		for (j = 0; j < out_cols; j++)
		{
			#pragma unroll
			for (c = 0; c < 4; c++)
				prod += shared_out_grad[j * 4 + c] * input_thread[j * strides * 4 + c];
		}
		input += cols * batch * strides;
		out_grad += out_cols * batch;
	}
	coeff[(channel_idx * count + count_idx) * filter_rows * filter_cols + origin_y * filter_cols + origin_x] = prod;
}
*/

template<int count_per_block, int batch_per_block>
__global__ static void _cwc_kern_convolutional_backward_propagate_coefficient_another(const int strides, const int border, const int batch,
		float* input, const int rows, const int cols, const int channels,
		float* out_grad, const int out_rows, const int out_cols,
		float* coeff, const int filter_rows, const int filter_cols, const int count)
{
	assert(gridDim.x == filter_cols);
	assert(gridDim.y == out_rows);
	extern __shared__ float shared[];
	float* shared_input = &shared[0];
	float* shared_out_grad = &shared[channels * batch_per_block];
	const int thidx = threadIdx.x + threadIdx.y * blockDim.x;
	const int thcnt = blockDim.x * blockDim.y;
	assert(thcnt >= channels * batch_per_block);
	assert(thcnt >= count);
	const int channel_idx = threadIdx.x;
	const int count_idx = threadIdx.y;
	const int origin_x = blockIdx.x;
	const int y = blockIdx.y;
	const int batch_group_count = batch / batch_per_block;
	const int start_x = max(origin_x - border, 0) - (origin_x - border);
	const int end_x = min(out_cols, (cols + border - origin_x + strides - 1) / strides);
	input += origin_x * channels * batch_per_block;
	out_grad += out_rows * out_cols * count * batch_per_block;
	int i, j, c, x;
	float prod[3][7];
	#pragma unroll
	for (i = 0; i < 3; i++)
		#pragma unroll
		for (j = 0; j < 7; j++)
			prod[i][j] = 0;
	const int iy = y * strides - border;
	for (x = start_x; x < end_x; x++)
	{
		if (thidx < channels * batch_per_block)
			#pragma unroll
			for (i = 0; i < 7; i++)
				shared_input[i * channels * batch_per_block + thidx] = (i + iy >= 0 && i + iy < rows) ? input[((i + y * strides - border) * cols + x * strides - border) * channels * batch_per_block + thidx] : 0;
		#pragma unroll
		for (i = 0; i < batch_per_block; i++)
		{
			shared_out_grad[thidx] = out_grad[(y * out_cols + x) * count * batch_per_block + i * count + thidx];
			__syncthreads();
			#pragma unroll
			for (c = 0; c < 7; c++)
				#pragma unroll
				for (j = 0; j < 3; j++)
					prod[c][j] += shared_out_grad[count_idx + j * count_per_block] * shared_input[c * channels * batch_per_block + i * channels + channel_idx];
			__syncthreads();
		}
	}
	#pragma unroll
	for (j = 0; j < 7; j++)
		#pragma unroll
		for (i = 0; i < 3; i++)
			coeff[(j * filter_cols + origin_x) * count * channels * batch_per_block + channel_idx * count + count_idx + i * count_per_block] = prod[i][j];
}

int main(int argc, char** argv)
{
	float* in = 0;
	float* out = 0;
	cudaMalloc(&in, sizeof(float) * (227 * 225 * 3 * 128));
	cudaMalloc(&out, sizeof(float) * (111 * 111 * 96 * 128));
	float* in_host = 0;
	float* out_host = 0;
	int i, j, c, k;
	cudaMallocHost(&in_host, sizeof(float) * 225 * 225 * 3 * 128);
	for (i = 0; i < 225; i++)
		for (j = 0; j < 225; j++)
			for (c = 0; c < 3; c++)
				for (k = 0; k < 128; k++)
					in_host[i * 225 * 3 * 128 + j * 3 * 128 + c * 128 + k] = c * k;
	cudaMemcpy(in, in_host, sizeof(float) * 225 * 225 * 3 * 128, cudaMemcpyHostToDevice);
	cudaMallocHost(&out_host, sizeof(float) * 111 * 111 * 96 * 128);
	for (i = 0; i < 111; i++)
		for (j = 0; j < 111; j++)
			for (c = 0; c < 96; c++)
				for (k = 0; k < 128; k++)
					out_host[i * 111 * 96 * 128 + j * 96 * 128 + c * 128 + k] = c * k;
	cudaMemcpy(out, out_host, sizeof(float) * 111 * 111 * 96 * 128, cudaMemcpyHostToDevice);
	float* w = 0;
	dim3 thread_per_block(96, 1);
	dim3 num_blocks(7, 7, 111 * 16);
	cudaMalloc(&w, sizeof(float) * 3 * 96 * 7 * 7 * 111 * 16);
	int shared_memory_size = sizeof(float) * 8 * (3 + 96);
	cudaFuncSetCacheConfig(_cwc_kern_convolutional_backward_propagate_coefficient_default<3, 1, 8>, cudaFuncCachePreferShared);
	_cwc_kern_convolutional_backward_propagate_coefficient_default
	<3, 1, 8>
	<<<num_blocks, thread_per_block, shared_memory_size>>>
	(2, 1, 128,
	 in, 225, 225, 3,
	 out, 111, 111,
	 w, 7, 7, 96);
	thread_per_block = dim3(3, 32);
	num_blocks = dim3(7, 111, 4);
	shared_memory_size = sizeof(float) * (32 * 7 * 3 + 96);
	cudaFuncSetCacheConfig(_cwc_kern_convolutional_backward_propagate_coefficient_another<32, 32>, cudaFuncCachePreferShared);
	_cwc_kern_convolutional_backward_propagate_coefficient_another
	<32, 32>
	<<<num_blocks, thread_per_block, shared_memory_size>>>
	(2, 1, 128,
	 in, 225, 225, 3,
	 out, 111, 111,
	 w, 7, 7, 96);
	/*
	thread_per_block = dim3(7, 7);
	num_blocks = dim3(3, 96, 32);
	shared_memory_size = sizeof(float) * 4 * (7 * (225 + 2) + 111);
	cudaFuncSetCacheConfig(_cwc_kern_convolutional_backward_propagate_coefficient_whole, cudaFuncCachePreferShared);
	_cwc_kern_convolutional_backward_propagate_coefficient_whole
	<<<num_blocks, thread_per_block, shared_memory_size>>>
	(2, 1, 128,
	 in, 225, 225, 3,
	 out, 111, 111,
	 w, 7, 7, 96);
	*/
	cudaFree(w);
	cudaFree(out);
	cudaFree(in);
	cudaFreeHost(out_host);
	cudaFreeHost(in_host);
	return 0;
}
