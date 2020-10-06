#include<iostream>
#include<cuda_runtime_api.h>
#include<time.h>
#include<stdlib.h>

#define SAFE_CALL(CallInstruction){ \
	cudaError_t cuerr=CallInstruction; \
	if(cuerr!=cudaSuccess){ \
		printf("CUDA error:%s at call \"" #CallInstruction"\"\n",cudaGetErrorString(cuerr));\
			throw "error in CUDA API function,aborting...";\
	} \
}

#define SAFE_KERNEL_CALL(KernelCallInstruction){\
	KernelCallInstruction; \
	cudaError_t cuerr=cudaGetLastError();\
	if(cuerr!=cudaSuccess){\
		printf("CUDA error in kernel launch:%s at kernel \"" #KernelCallInstruction "\"\n",cudaGetErrorString(cuerr)); \
			throw "error in CUDA kernel launch,aborting...";\
	}\
	cuerr=cudaDeviceSynchronize();\
	if(cuerr!=cudaSuccess){\
		printf("CUDA error in kernel execution:%s at kernel\"" #KernelCallInstruction "\"\n",cudaGetErrorString(cuerr));\
			throw "error in CUDA kernel execution,aborting...";\
	}\
}

#define size_mb 1048576

//ядро
__global__ void ram(int *ptrs,int *result,int *data,int num)
{
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if(idx<num)
	{
		result[idx]=data[ptrs[idx]];
	}
}


int main(int argc,char **argv)
{
	int n=atoi(argv[1]);
	long int cached_array_size=size_mb*n;
	long int large_array_size=size_mb*1024;

	//host data
	int *h_ptrs,*h_result,*h_data;

	h_ptrs=(int*)malloc(large_array_size);	
	h_result=(int*)malloc(large_array_size);
	h_data=(int*)malloc(cached_array_size);

	for(int i=0;i<cached_array_size;i++)
	{
		h_data[i]=i;
	}

	for(int i=0;i<large_array_size;i++)
	{
		h_result[i]=0;
		h_ptrs[i]=rand()%cached_array_size;
	}

	//device data
	int *d_ptrs,*d_result,*d_data;

	SAFE_CALL(cudaMalloc(&d_ptrs,large_array_size));
	SAFE_CALL(cudaMalloc(&d_result,large_array_size));
	SAFE_CALL(cudaMalloc(&d_data,cached_array_size));

	SAFE_CALL(cudaMemcpy(d_ptrs,h_ptrs,large_array_size,cudaMemcpyHostToDevice));
	SAFE_CALL(cudaMemcpy(d_result,h_result,large_array_size,cudaMemcpyHostToDevice));
	SAFE_CALL(cudaMemcpy(d_data,h_data,cached_array_size,cudaMemcpyHostToDevice));
	int blocksize=1024;
	int gridsize=(((large_array_size/sizeof(int))-1)/blocksize)+1;

	cudaEvent_t start,finish;
	SAFE_CALL(cudaEventCreate(&start));
	SAFE_CALL(cudaEventCreate(&finish));
	
	SAFE_CALL(cudaDeviceSynchronize());

	SAFE_CALL(cudaEventRecord(start));
	SAFE_KERNEL_CALL((ram<<<gridsize,blocksize>>>(d_ptrs,d_result,d_data,large_array_size/sizeof(int))));
	SAFE_CALL(cudaEventRecord(finish));

	SAFE_CALL(cudaMemcpy(h_result,d_result,large_array_size,cudaMemcpyDeviceToHost));
	SAFE_CALL(cudaEventSynchronize(finish));
	
	float milliseconds=0;
	SAFE_CALL(cudaEventElapsedTime(&milliseconds,start,finish));
	
	printf("Used time is(s):%f",milliseconds/1000);
	printf("Effective Bandwidth(GB/s):%fn",large_array_size*3/milliseconds/1e6);
	
	FILE *bandwidth=fopen("bandwidth.dat","a");
	fprintf(bandwidth,"%d %fn",n,large_array_size*3/milliseconds/1e6);

	SAFE_CALL(cudaFree(d_ptrs));
	SAFE_CALL(cudaFree(d_result));
	SAFE_CALL(cudaFree(d_data));

	free(h_ptrs);
	free(h_result);
	free(h_data);

	return 0;
}
