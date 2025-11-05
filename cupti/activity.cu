#include <cuda_runtime.h>
#include <stdio.h>
#include <cupti.h>

// Callback for buffer requests
static void BufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
    *size = 8 * 1024 * 1024; // 8MB buffer
    *maxNumRecords = 0;
    *buffer = (uint8_t*)malloc(*size);
}

// Callback for buffer completed
static void BufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t* buffer, size_t size, size_t validSize) {
    CUpti_Activity *record = NULL;

    if (validSize > 0)
    {
        // Parse CUPTI activity records here, print kernel name and duration
        while (cuptiActivityGetNextRecord(buffer, validSize, &record) == CUPTI_SUCCESS)
        {
            if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
                CUpti_ActivityKernel8 *kernel = (CUpti_ActivityKernel8 *)record;
                printf("\n[Kernel]\n");
		printf("kernel name = %s\n", kernel->name);
                printf("kernel duration (ns) = %llu\n", (unsigned long long)(kernel->end - kernel->start));
            }else if( record->kind == CUPTI_ACTIVITY_KIND_MEMCPY){
		    CUpti_ActivityMemcpy *memcpy = (CUpti_ActivityMemcpy *)record;

    const char *copyKindStr = "";
    switch (memcpy->copyKind) {
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
            copyKindStr = "Host → Device";
            break;
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
            copyKindStr = "Device → Host";
            break;
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
            copyKindStr = "Device → Device";
            break;
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
            copyKindStr = "Host → Array";
            break;
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
            copyKindStr = "Array → Host";
            break;
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
            copyKindStr = "Array → Array";
            break;
        default:
            copyKindStr = "Other";
            break;
    }

    printf("\n[Memcpy]\n");
    printf("  Kind: %s\n", copyKindStr);
    printf("  Bytes: %llu\n", (unsigned long long)memcpy->bytes);
    printf("  Duration (ns): %llu\n",
           (unsigned long long)(memcpy->end - memcpy->start));
    printf("  DeviceId: %u, StreamId: %u\n",
           memcpy->deviceId, memcpy->streamId);
	    }
        }
    }
    free(buffer);
}


// CUDA kernel for vector addition
__global__ void VectorAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

int main() {
    int vectorLen = 1024 * 1024;
    size_t size = vectorLen * sizeof(float);

    // Step 1: Register CUPTI callbacks
    cuptiActivityRegisterCallbacks(BufferRequested, BufferCompleted);

    // Step 2: Enable CUPTI Activity Collection
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);

    // Host memory allocation
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize vectors
    for (int i = 0; i < vectorLen; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Device memory allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 128;
    int blocksPerGrid = (vectorLen + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel (profiler will capture this call)
    VectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, vectorLen);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    // Step 3: Flushing and Disabling CUPTI Activity
    cuptiActivityFlushAll(1);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);

    return 0;
}
