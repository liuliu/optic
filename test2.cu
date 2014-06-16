#include <cuda.h>
#include <assert.h>
#include <stdio.h>

template <int input_per_thread, int filter_per_thread, int input_per_block, int filter_per_block>
__global__ static void _cwc_kern_convolutional_forward_propagate(const int strides, const int border, const int batch,
		float* input, const int rows, const int cols, const int channels_per_partition, const int partition,
		float* out, const int out_rows, const int out_cols,
		float* filter, const int filter_rows, const int filter_cols, const int count,
		float* const biases)
{
	assert(gridDim.x * partition * filter_per_block * input_per_block == out_cols * batch * count);
	assert(gridDim.y == out_rows);
	assert(gridDim.z == partition);
	extern __shared__ float shared[];
	float* shared_block = &shared[0];
	float* shared_weights = &shared[input_per_block];
	float* shared_bias = &shared[input_per_block + filter_per_block];
	float prod[filter_per_thread][input_per_thread];
	assert(input_per_block == input_per_thread * blockDim.x);
	assert(filter_per_block == filter_per_thread * blockDim.y);
	const int thidx = threadIdx.x + threadIdx.y * blockDim.x;
	const int thcnt = blockDim.x * blockDim.y;
	int c, i, j, x, y;
	#pragma unroll
	for (i = 0; i < filter_per_thread; i++)
		#pragma unroll
		for (j = 0; j < input_per_thread; j++)
		prod[i][j] = 0;
	const int origin_x = blockIdx.x % out_cols;
	const int origin_y = blockIdx.y;
	const int input_group_count = batch / input_per_block;
	const int input_group_idx = (blockIdx.x % (out_cols * input_group_count)) / out_cols;
	const int filter_group_idx = blockIdx.z * count / (filter_per_block * partition) + blockIdx.x / (out_cols * input_group_count); // for the partitioned filter group
	input += (blockIdx.z * channels_per_partition * rows * cols +  origin_y * strides * cols + origin_x * strides) * batch + input_group_idx * input_per_block;
	assert(thcnt >= input_per_block);
	assert(thcnt >= filter_per_block);
	if (thidx < filter_per_block)
		shared_bias[thidx] = biases[filter_group_idx * filter_per_block + thidx];
	const int start_x = max(origin_x * strides - border, 0) - (origin_x * strides - border);
	const int end_x = min(origin_x * strides - border + filter_cols, cols) - (origin_x * strides - border);
	const int start_y = max(origin_y * strides - border, 0) - (origin_y * strides - border);
	const int end_y = min(origin_y * strides - border + filter_rows, rows) - (origin_y * strides - border);
	filter += filter_group_idx * filter_per_block;
	for (c = 0; c < channels_per_partition; c++)
	{
		for (y = start_y; y < end_y; y++)
			for (x = start_x; x < end_x; x++)
			{
				if (thidx < input_per_block)
					shared_block[thidx] = input[((y - border) * cols + x - border) * batch + thidx];
				if (thidx < filter_per_block)
					shared_weights[thidx] = filter[(y * filter_cols + x) * count + thidx];
				__syncthreads();
				#pragma unroll
				for (i = 0; i < filter_per_thread; i++)
					#pragma unroll
					for (j = 0; j < input_per_thread; j++)
						prod[i][j] += shared_block[j + threadIdx.x * input_per_thread] * shared_weights[i + threadIdx.y * filter_per_thread];
				__syncthreads();
			}
		input += rows * cols * batch;
		filter += filter_rows * filter_cols * count;
	}
	const int outcnt = out_rows * out_cols * batch;
	out += (filter_group_idx * filter_per_block + threadIdx.y * filter_per_thread) * outcnt + (origin_y * out_cols + origin_x) * batch + input_group_idx * input_per_block + threadIdx.x * input_per_thread;
	#pragma unroll
	for (i = 0; i < filter_per_thread; i++)
	{
		const float bias = shared_bias[i + threadIdx.y * filter_per_thread];
		#pragma unroll
		for (j = 0; j < input_per_thread; j++)
			out[j] = max(0.0, prod[i][j] + bias);
		out += outcnt;
	}
}

int main(int argc, char** argv)
{
	float* in = 0;
	float* out = 0;
	cudaMalloc(&in, sizeof(float) * (55 * 55 * 96 * 256));
	cudaMalloc(&out, sizeof(float) * (27 * 27 * 256 * 256));
	float* in_host = 0;
	float* out_host = 0;
	int i, j, c, k;
	cudaMallocHost(&in_host, sizeof(float) * 55 * 55 * 96 * 128);
	for (i = 0; i < 55; i++)
		for (j = 0; j < 55; j++)
			for (c = 0; c < 96; c++)
				for (k = 0; k < 128; k++)
					in_host[i * 55 * 96 * 128 + j * 96 * 128 + c * 128 + k] = c * k;
	cudaMemcpy(in, in_host, sizeof(float) * 55 * 55 * 96 * 128, cudaMemcpyHostToDevice);
	cudaMallocHost(&out_host, sizeof(float) * 27 * 27 * 256 * 128);
	for (i = 0; i < 27; i++)
		for (j = 0; j < 27; j++)
			for (c = 0; c < 256; c++)
				for (k = 0; k < 128; k++)
					out_host[i * 27 * 256 * 128 + j * 256 * 128 + c * 128 + k] = c * k;
	cudaMemcpy(out, out_host, sizeof(float) * 27 * 27 * 256 * 128, cudaMemcpyHostToDevice);
	float* w = 0;
	cudaMalloc(&w, sizeof(float) * (256 * 96 / 2) * 5 * 5);
	float* biases = 0;
	cudaMalloc(&biases, sizeof(float) * 256);
	dim3 thread_per_block(64 / 4, 32 / 8);
	dim3 num_blocks(27 * 2 * 256 / (32 * 2), 27, 2);
	int shared_memory_size = sizeof(float) * (64 + 32 * 2);
	cudaFuncSetCacheConfig(_cwc_kern_convolutional_forward_propagate<4, 8, 64, 32>, cudaFuncCachePreferL1);
	_cwc_kern_convolutional_forward_propagate
	<4, 8, 64, 32>
	<<<num_blocks, thread_per_block, shared_memory_size>>>
	(2, 1, 128,
	 in, 55, 55, 96 / 2, 2,
	 out, 27, 27,
	 w, 5, 5, 256,
	 biases);
	cudaFree(biases);
	cudaFree(w);
	cudaFree(out);
	cudaFree(in);
	cudaFreeHost(out_host);
	cudaFreeHost(in_host);
	return 0;
}
