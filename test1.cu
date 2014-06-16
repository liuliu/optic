#include <cuda.h>
#include <assert.h>
#include <stdio.h>

template <int channel_per_thread, int filter_per_thread, int channel_per_block, int filter_per_block, int batch_per_block>
__global__ static void _cwc_kern_convolutional_backward_propagate_coefficient_default(const int strides, const int border, const int batch, const int batch_group_count,
		float* input, const int rows, const int cols, const int channels_per_partition, const int partition,
		float* out_grad, const int out_rows, const int out_cols,
		float* coeff, const int filter_rows, const int filter_cols, const int count_per_partition)
{
	assert(gridDim.x == strides);
	assert(gridDim.y == filter_rows);
	assert(gridDim.z * channel_per_block * filter_per_block * batch_per_block == count_per_partition * channels_per_partition * partition * batch);
	assert(batch == batch_per_block * batch_group_count);
	extern __shared__ float shared[];
	float* shared_input = &shared[0];
	float* shared_out_grad = &shared[channel_per_block * 3];
	const int thidx = threadIdx.x + threadIdx.y * blockDim.x;
	const int thcnt = blockDim.x * blockDim.y;
	assert(blockDim.x * filter_per_thread == filter_per_block);
	assert(blockDim.y * channel_per_thread == channel_per_block);
	assert(thcnt >= channel_per_block);
	assert(thcnt >= filter_per_block);
	const int origin_x = blockIdx.x;
	const int origin_y = blockIdx.y;
	const int channel_group_count = channels_per_partition / channel_per_block;
	const int filter_group_count = count_per_partition / filter_per_block;
	const int partition_idx = blockIdx.z / (channel_group_count * filter_group_count * batch_group_count);
	const int batch_group_idx = (blockIdx.z % (channel_group_count * filter_group_count * batch_group_count)) / (channel_group_count * filter_group_count);
	const int filter_group_idx = (blockIdx.z % (channel_group_count * filter_group_count)) / channel_group_count;
	const int channel_group_idx = blockIdx.z % channel_group_count;
	const int start_x = max(origin_x - border, 0) - (origin_x - border);
	const int end_x = min(out_cols, (cols + border - origin_x + strides - 1) / strides);
	const int start_y = max(origin_y - border, 0) - (origin_y - border);
	const int end_y = min(out_rows, (rows + border - origin_y + strides - 1) / strides);
	input += (partition_idx * batch + batch_group_idx * batch_per_block) * rows * cols * channels_per_partition + (origin_y * cols + origin_x) * channels_per_partition + channel_group_idx * channel_per_block;
	out_grad += (partition_idx * batch + batch_group_idx * batch_per_block) * out_rows * out_cols * count_per_partition + filter_group_idx * filter_per_block;
	int i, j, k, c, x, y;
	float prod[channel_per_thread][filter_per_thread][3];
	#pragma unroll
	for (i = 0; i < channel_per_thread; i++)
		#pragma unroll
		for (j = 0; j < filter_per_thread; j++)
			#pragma unroll
			for (k = 0; k < 3; k++)
				prod[i][j][k] = 0;
	__syncthreads();
	for (c = 0; c < batch_per_block; c++)
	{
		for (y = start_y; y < end_y; y++)
		{
			#pragma unroll
			for (k = 0; k < 2; k++)
				if (thidx < channel_per_block)
					shared_input[k * channel_per_block + thidx] = k * strides - border + origin_x < cols ? input[((y * strides - border) * cols + k * strides - border) * channels_per_partition + thidx] : 0;
			for (x = start_x; x < end_x; x++)
			{
				if (thidx < filter_per_block)
					shared_out_grad[thidx] = out_grad[(y * out_cols + x) * count_per_partition + thidx];
				if (thidx < channel_per_block)
					shared_input[((x - start_x + 2) % 3) * channel_per_block + thidx] = (x + 2) * strides - border + origin_x < cols ? input[((y * strides - border) * cols + (x + 2) * strides - border) * channels_per_partition + thidx] : 0;
				__syncthreads();
				#pragma unroll
				for (k = 0; k < 3; k++)
					#pragma unroll
					for (i = 0; i < channel_per_thread; i++)
						#pragma unroll
						for (j = 0; j < filter_per_thread; j++)
							prod[i][j][k] += shared_input[((x - start_x + k) % 3) * channel_per_block + i + threadIdx.y * channel_per_thread] * shared_out_grad[j + threadIdx.x * filter_per_thread];
				__syncthreads();
			}
		}
		input += rows * cols * channels_per_partition;
		out_grad += out_rows * out_cols * count_per_partition;
	}
	const int cocnt = filter_cols * filter_rows * count_per_partition * partition;
	coeff += cocnt * (channels_per_partition * batch_group_idx + channel_group_idx * channel_per_block) + (origin_y * filter_cols + origin_x) * count_per_partition * partition + partition_idx * count_per_partition + filter_group_idx * filter_per_block;
	#pragma unroll
	for (k = 0; k < 3; k++)
		if (k * strides + origin_x < filter_cols)
			#pragma unroll
			for (i = 0; i < channel_per_thread; i++)
				#pragma unroll
				for (j = 0; j < filter_per_thread; j++)
						coeff[(i + threadIdx.y * channel_per_thread) * cocnt + k * strides * count_per_partition * partition + j + threadIdx.x * filter_per_thread] = prod[i][j][k];
}

int main(int argc, char** argv)
{
	float* in = 0;
	float* out = 0;
	cudaMalloc(&in, sizeof(float) * (55 * 55 * 96 * 128));
	cudaMalloc(&out, sizeof(float) * (27 * 27 * 256 * 128));
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
	dim3 thread_per_block(128 / 4, 16 / 4);
	dim3 num_blocks(2, 5, (96 / 2 / 16) * (256 / 2 / 128) * 2 * 16);
	cudaMalloc(&w, sizeof(float) * (256 * 96 / 2) * 5 * 5 * 16);
	int shared_memory_size = sizeof(float) * (16 * 5 + 128);
	cudaFuncSetCacheConfig(_cwc_kern_convolutional_backward_propagate_coefficient_default<4, 4, 16, 128, 8>, cudaFuncCachePreferShared);
	_cwc_kern_convolutional_backward_propagate_coefficient_default
	<4, 4, 16, 128, 8>
	<<<num_blocks, thread_per_block, shared_memory_size>>>
	(2, 1, 128, 16,
	 in, 55, 55, 96 / 2, 2,
	 out, 27, 27,
	 w, 5, 5, 256 / 2);
	cudaFree(w);
	cudaFree(out);
	cudaFree(in);
	cudaFreeHost(out_host);
	cudaFreeHost(in_host);
	return 0;
}
