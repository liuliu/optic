#include <cuda.h>
#include <assert.h>
#include <stdio.h>

// this method rewinds a matrix
template <int parallel_threads>
__global__ static void _cwc_kern_reorder_matrix_major(float* a, float* b, const int count, const int channels_per_partition, const int partition, const int batch)
{
	assert(blockDim.x == parallel_threads);
	const int batch_group_idx = blockIdx.y % (batch / parallel_threads);
	const int channel_group_idx = blockIdx.y / (batch / parallel_threads);
	a += (blockIdx.z * count * channels_per_partition + blockIdx.x + channel_group_idx * parallel_threads * count) * batch + batch_group_idx * parallel_threads;
	b += (blockIdx.z * count * batch + batch_group_idx * parallel_threads * count + blockIdx.x) * channels_per_partition + channel_group_idx * parallel_threads;
	__shared__ float prod[parallel_threads][parallel_threads];
	int i;
	#pragma unroll
	for (i = 0; i < parallel_threads; i++)
		prod[i][threadIdx.x] = a[i * count * batch + threadIdx.x];
	__syncthreads();
	#pragma unroll
	for (i = 0; i < parallel_threads; i++)
		b[i * count * channels_per_partition + threadIdx.x] = prod[threadIdx.x][i];
	__syncthreads();
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
	float* chin = 0;
	float* chout = 0;
	cudaMalloc(&chin, sizeof(float) * (55 * 55 * 96 * 256));
	cudaMalloc(&chout, sizeof(float) * (27 * 27 * 256 * 256));
	_cwc_kern_reorder_matrix_major
	<16>
	<<<dim3(55 * 55, (96 / 2 / 16) * (128 / 16), 2), 16, 16 * 16 * sizeof(float)>>>
	(in, chin, 55 * 55, 96 / 2, 2, 128);
	_cwc_kern_reorder_matrix_major
	<16>
	<<<dim3(27 * 27, (256 / 2 / 16) * (128 / 16), 2), 16, 16 * 16 * sizeof(float)>>>
	(out, chout, 27 * 27, 256 / 2, 2, 128);
	cudaFree(out);
	cudaFree(in);
	cudaFreeHost(out_host);
	cudaFreeHost(in_host);
	return 0;
}
