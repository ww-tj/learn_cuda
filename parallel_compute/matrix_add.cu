#include<stdio.h>
#include<stdlib.h>


void printMatrix(float *, int);

__global__ void addMatrix(float *a, float *b, float *c, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        c[i] = a[i] + b[i];
    }
}


int main(){
    int N = 5;
    size_t SIZE = sizeof(float) * N;
    float *h_A = (float *) malloc(SIZE);
    float *h_B = (float *) malloc(SIZE);
    float *h_C = (float *) malloc(SIZE);

    for(int i = 0; i < N; i++){
        h_A[i] = (float) rand() / RAND_MAX;
        h_B[i] = (float) rand() / RAND_MAX;
    }


    float *d_A = NULL, *d_B = NULL, *d_C =NULL;
    cudaMalloc((void **) &d_A, SIZE);
    cudaMalloc((void **) &d_B, SIZE);
    cudaMalloc((void **) &d_C, SIZE);

    cudaMemcpy(d_A, h_A, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, SIZE, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((N + threadsPerBlock.x -1)/threadsPerBlock.x);
    addMatrix<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, SIZE, cudaMemcpyDeviceToHost);
    printMatrix(h_A, N);
    printMatrix(h_B, N);
    printMatrix(h_C, N);

    free(h_A);
    free(h_B);
    free(h_C);


    return 0;
}

void printMatrix(float *a, int N){
    for(int i = 0; i < N; i++)
        printf("%.3f ", a[i]);
    printf("\n");
}