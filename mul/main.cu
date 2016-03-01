#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <assert.h>

#define BILLION 1000000000L
#define E(m,n,row,col) (m[n * row + col])



__device__ __host__ void matrix_init_serial(float * a, int n)
{
    int i, j;
    for(i = 0; i < n; i++)
        for(j = 0; j < n; j++)
            E(a,n,i,j) = (float)i * j;
}

__device__ __host__ void matrix_mul_serial(float * dest, float * a, float * b,int n)
{
    int i, j, k;
    float sum;
    for(i = 0; i < n; i++)
    {
        for(j = 0; j < n; j++)
        {
            sum = 0.f;
            for(k = 0; k < n; k++)
            {
                sum += (E(a,n,i,j)*E(b,n,i,j));
            }
            E(dest,n,i,j) = sum;
        }
    }
}



__global__ void matrix_mul_parallel_gpu(float * dest, float * a, float * b,int n){
    int k;
    float sum = 0.f;
    for(k = 0; k < n; k++)
    {
        sum += (E(a,blockDim.x,blockIdx.x,threadIdx.x) *
                E(b,blockDim.x,blockIdx.x,threadIdx.x));
    }

    E(dest, blockDim.x, blockIdx.x, threadIdx.x) = sum;
}

__global__ void matrix_init_parallel_gpu(float * a, int n)
{
    E(a, blockDim.x, blockIdx.x, threadIdx.x) =
        blockIdx.x * threadIdx.x;
}
#define HANDLE_ERROR(e)  _HANDLE_ERROR(e, __LINE__)

void _HANDLE_ERROR(cudaError_t e, int line)
{
    if (e != cudaSuccess)
    {
        printf("line: %d. error %s\n", line, cudaGetErrorString(e));
        exit (1);
    }
}

int main(int argc, char * argv[])
{
    if (argc != 2)
    {
        fprintf(stderr,"usage: %s <matrix-size>\n", argv[0]);
        return 1;
    }
    struct  timespec    start,  end;
    uint64_t diff1, diff3;
    float * dev_a, * dev_b, * dev_c, * dev_d;
    int i,j;
    int N = atoi(argv[1]);
    float * c = (float*)malloc(sizeof(float) * N * N);
    float * d = (float*)malloc(sizeof(float) * N * N);
    float * cpua = (float*)malloc(sizeof(float) * N * N);
    float * cpub = (float*)malloc(sizeof(float) * N * N);
    float * cpuc = (float*)malloc(sizeof(float) * N * N);

    // allocate the memory on the GPU
    HANDLE_ERROR( cudaMalloc( (void**)&dev_a, N*N * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_b, N*N * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_c, N*N * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_d, N*N * sizeof(float) ) );

    clock_gettime(CLOCK_MONOTONIC, &start);
    {
        matrix_init_serial( cpua, N);
        matrix_init_serial( cpub, N);
        matrix_mul_serial( cpuc, cpua, cpub, N);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    diff3 = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
    printf("Elapsed time cpu   =   %llu    ns\n",  (uint64_t)diff3);

    clock_gettime(CLOCK_MONOTONIC, &start);
    {
        matrix_init_parallel_gpu<<<N,N>>>( dev_a, N);
        matrix_init_parallel_gpu<<<N,N>>>( dev_b, N);
        matrix_mul_parallel_gpu<<<N,N>>>( dev_c, dev_a, dev_b, N);

        HANDLE_ERROR( cudaMemcpy( c, dev_c, N *N* sizeof(float),
                    cudaMemcpyDeviceToHost) );
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    diff1 = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
    printf("Elapsed time gpu parallel   =   %llu    ns\n",  (uint64_t)diff1);



#ifdef GPU_SERIAL
    uint64_t diff2;
    clock_gettime(CLOCK_MONOTONIC, &start);
    {

        matrix_init_serial<<<1,1>>>( dev_a, N);
        matrix_init_serial<<<1,1>>>( dev_b, N);
        matrix_mul_serial<<<1,1>>>( dev_d, dev_a, dev_b, N);

        HANDLE_ERROR( cudaMemcpy( d, dev_b, N *N* sizeof(float),
                cudaMemcpyDeviceToHost) );
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    diff2 = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
    printf("Elapsed time gpu serial   =   %llu    ns\n",  (uint64_t)diff2);
#endif

#define CHECK_CORRECTNESS
#ifdef CHECK_CORRECTNESS
#define MAX(a,b) (a < b ? b : a)
#define ACCEPTABLE(a,b) (a==b ? 1 :MAX(a/b,b/a) < 1.0+1e-4)
    for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
#ifdef GPU_SERIAL
            assert(ACCEPTABLE(E(c,N, i,j), E(d,N, i,j)));
            assert(ACCEPTABLE(E(cpuc,N, i,j),E(d,N, i,j)));
#endif

            if(!ACCEPTABLE(E(c,N, i,j), E(cpuc,N, i,j)))
            {
                printf("%f   %f\n",E(c,N, i,j), E(cpuc,N, i,j));
            }
            assert(ACCEPTABLE(E(c,N, i,j), E(cpuc,N, i,j)));
        }
    }
#endif

#ifdef GPU_SERIAL
    printf("speedup serial gpu = %.2fx\n", (float)diff2 / (float)diff1);
#endif
    printf("speedup on cpu = %.2fx\n", (float)diff3 / (float)diff1);

    cudaFree( dev_a );
    cudaFree( dev_b );
    cudaFree( dev_c );
    free(c);
    free(d);
    free(cpua);
    free(cpub);
    free(cpuc);
    
    return 0;
}
