#include <cuda.h>
#include <assert.h>
#include <stdio.h>

// this method rewinds a matrix
template <int batch_per_block>
__global__ static void _cwc_kern_reorder_matrix_major_per_block_rows(float* a, float* b, const int count, const int channels, const int batch)
{
	const int thidx = blockIdx.y * batch_per_block + threadIdx.y;
	b[(blockIdx.y * count + blockIdx.x) * channels * batch_per_block + threadIdx.y * channels + threadIdx.x] = a[(threadIdx.x * count + blockIdx.x) * batch + thidx];
}
// this method rewinds a matrix
template <int channel_per_block, int batch_per_block, int batch_group_per_block>
__global__ static void _cwc_kern_reorder_matrix_major_per_block(float* a, float* b, const int count, const int channels, const int batch)
{
	const int batch_group_idx = blockIdx.y % (batch / (batch_per_block * batch_group_per_block));
	const int channel_group_idx = blockIdx.y / (batch / (batch_per_block * batch_group_per_block));
	a += (channel_group_idx * channel_per_block * count + blockIdx.x) * batch + batch_group_idx * batch_per_block * batch_group_per_block;
	b += (batch_group_idx * batch_group_per_block * count + blockIdx.x) * channels * batch_per_block + channel_group_idx * channel_per_block;
	__shared__ float prod[channel_per_block][batch_per_block * batch_group_per_block];
	int i, j;
	#pragma unroll
	for (i = 0; i < channel_per_block; i++)
		prod[i][threadIdx.x] = a[i * count * batch + threadIdx.x];
	__syncthreads();
	if (threadIdx.x < channel_per_block)
		#pragma unroll
		for (i = 0; i < batch_group_per_block; i++)
			#pragma unroll
			for (j = 0; j < batch_per_block; j++)
				b[(i * count * batch_per_block + j) * channels + threadIdx.x] = prod[threadIdx.x][i * batch_per_block + j];
}

int main(int argc, char** argv)
{
	float* in = 0;
	float* out = 0;
	cudaMalloc(&in, sizeof(float) * (225 * 225 * 3 * 256));
	cudaMalloc(&out, sizeof(float) * (111 * 111 * 96 * 256));
	float* in_host = 0;
	float* out_host = 0;
	int i, j, c, k;
	cudaMallocHost(&in_host, sizeof(float) * 225 * 225 * 3 * 256);
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
	float* chin = 0;
	float* chout = 0;
	cudaMalloc(&chin, sizeof(float) * (225 * 225 * 3 * 256));
	cudaMalloc(&chout, sizeof(float) * (111 * 111 * 96 * 256));
	_cwc_kern_reorder_matrix_major_per_block_rows
	<8>
	<<<dim3(225 * 225, 128 / (8 * 2)), dim3(3, 8)>>>
	(in, chin, 225 * 225, 3, 128);
	_cwc_kern_reorder_matrix_major_per_block
	<3, 8, 2>
	<<<dim3(225 * 225, 128 / (8 * 2)), 16, sizeof(float) * 3 * 8 * 2>>>
	(in, chin, 225 * 225, 3, 128);
	_cwc_kern_reorder_matrix_major_per_block
	<16, 8, 2>
	<<<dim3(111 * 111, (96 / 16) * (128 / (8 * 2))), 16, sizeof(float) * 16 * 8 * 2>>>
	(out, chout, 111 * 111, 96, 128);
	cudaFree(out);
	cudaFree(in);
	cudaFreeHost(out_host);
	cudaFreeHost(in_host);
	return 0;
}
