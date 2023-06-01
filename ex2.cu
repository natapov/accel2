#include "ex2.h"
#include <cuda/atomic>
#include <queue>
using namespace std;

#define NUM_STREAMS 64
#define QUE_MAX 16
const int img_size = SIZE * SIZE * CHANNELS;

// Our functions from ex 1

__device__ void prefixSum(int arr[], int size, int tid, int threads) {
    int increment;
    const auto is_active = tid < size;
    for (int stride = 1; stride<size; stride*=2) {
        if (tid >= stride && is_active) {
            increment = arr[tid - stride];
        }
        __syncthreads();
        if (tid >= stride && is_active) {
            arr[tid] += increment;
        }
        __syncthreads();
    }
}

__device__ void argmin(int arr[], int len, int tid, int threads) {
    int halfLen = len / 2;
    bool firstIteration = true;
    int prevHalfLength = 0;
    while (halfLen > 0) {
        if(tid < halfLen){
            if(arr[tid] == arr[tid + halfLen]){ //a corenr case
                int lhsIdx = tid;
                int rhdIdx = tid + halfLen;
                int lhsOriginalIdx = firstIteration ? lhsIdx : arr[prevHalfLength + lhsIdx];
                int rhsOriginalIdx = firstIteration ? rhdIdx : arr[prevHalfLength + rhdIdx];
                arr[tid + halfLen] = lhsOriginalIdx < rhsOriginalIdx ? lhsOriginalIdx : rhsOriginalIdx;
            }
            else { //the common case
                bool isLhsSmaller = (arr[tid] < arr[tid + halfLen]);
                int idxOfSmaller = isLhsSmaller * tid + (!isLhsSmaller) * (tid + halfLen);
                int smallerValue = arr[idxOfSmaller];
                int origIdxOfSmaller = firstIteration * idxOfSmaller + (!firstIteration) * arr[prevHalfLength + idxOfSmaller];
                arr[tid] = smallerValue;
                arr[tid + halfLen] = origIdxOfSmaller;
            }
        }
        __syncthreads();
        firstIteration = false;
        prevHalfLength = halfLen;
        halfLen /= 2;
    }
}

__device__ void zero_array(int* histograms, int size=CHANNELS*LEVELS) {
    auto hist_flat = (int*) histograms;
    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    for(int i = tid; i < size; i+=threads) {
        hist_flat[i] = 0;
    }
}

__device__ void test_array(uchar* arr, int size) {
    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    for(int i = tid; i < size; i+=threads) {
        arr[i] = arr[i];
    }
}


__device__ void colorHist(uchar img[][CHANNELS], int histograms[][LEVELS]) {
    const int pic_size = SIZE * SIZE;
    const int tid = threadIdx.x;
    const int threads = blockDim.x;

    for (int i = tid; i < 3*pic_size; i+=threads) {
        const int color = i%3;
        const int pixel = i/3;
        atomicAdd(&histograms[color][img[pixel][color]], 1);
    }
}


__device__ void performMapping(int maps[][LEVELS], uchar targetImg[][CHANNELS], uchar resultImg[][CHANNELS]){
    int pixels = SIZE * SIZE;
    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    for (int i = tid; i < pixels; i+= threads) {
        uchar *inRgbPixel = targetImg[i];
        uchar *outRgbPixel = resultImg[i];
        for (int j = 0; j < CHANNELS; j++){
            int *mapChannel = maps[j];
            outRgbPixel[j] = mapChannel[inRgbPixel[j]];
        }
    }    
}
__global__
void process_image_kernel(uchar *targets, uchar *references, uchar *results) {
    int tid = threadIdx.x;;
    int threads = blockDim.x;
    int bid = blockIdx.x;
    __shared__ int deleta_cdf_row[LEVELS];
    __shared__ int map_cdf[CHANNELS][LEVELS];
    __shared__ int histogramsShared_target[CHANNELS][LEVELS];
    __shared__ int histogramsShared_refrence[CHANNELS][LEVELS];
    zero_array((int*)histogramsShared_target,   CHANNELS * LEVELS);
    zero_array((int*)histogramsShared_refrence, CHANNELS * LEVELS);
    zero_array((int*)map_cdf,                   CHANNELS * LEVELS);
    zero_array((int*)deleta_cdf_row,            LEVELS);

    auto target   = (uchar(*)[CHANNELS]) &targets[  bid * img_size];
    auto reference = (uchar(*)[CHANNELS]) &references[bid * img_size];
    auto result   = (uchar(*)[CHANNELS]) &results[  bid * img_size];

    colorHist(target, histogramsShared_target);
    colorHist(reference, histogramsShared_refrence);
    __syncthreads();

    for(int c=0; c < CHANNELS; c++)
    {   
        prefixSum(histogramsShared_target[c],LEVELS, threadIdx.x, blockDim.x);
        prefixSum(histogramsShared_refrence[c], LEVELS, threadIdx.x, blockDim.x);
        __syncthreads();

        for (int i = 0; i < LEVELS; i+=1) {
            for (int j = tid; j < LEVELS; j+=threads) {
                deleta_cdf_row[j] = abs(histogramsShared_target[c][i]-histogramsShared_refrence[c][j]);
            }
            __syncthreads();
            argmin(deleta_cdf_row, LEVELS, threadIdx.x, blockDim.x);
            __syncthreads();

            map_cdf[c][i] = deleta_cdf_row[1];

            __syncthreads();
        }
        __syncthreads();
    }          

    //Preform Map
    performMapping(map_cdf, target, result); 
    __syncthreads(); 
}

// Our functions from ex 1 end

__device__ 
void process_image(uchar *targets, uchar *references, uchar *results, int deleta_cdf_row[LEVELS], int map_cdf[][LEVELS], int histogramsShared_target[][LEVELS], int histogramsShared_refrence[][LEVELS]) {
    int tid = threadIdx.x;;
    int threads = blockDim.x;
    int bid = blockIdx.x;
    assert(bid==0);
    assert(tid < 256 );
    assert(targets);
    assert(references);
    assert(results);
    zero_array((int*)histogramsShared_target,   CHANNELS * LEVELS);
    zero_array((int*)histogramsShared_refrence, CHANNELS * LEVELS);
    zero_array((int*)map_cdf,                   CHANNELS * LEVELS);
    zero_array((int*)deleta_cdf_row,            LEVELS);

    auto target   = (uchar(*)[CHANNELS]) &targets   [bid * img_size];
    auto refrence = (uchar(*)[CHANNELS]) &references[bid * img_size];
    auto result   = (uchar(*)[CHANNELS]) &results   [bid * img_size];

    colorHist(target, histogramsShared_target);
    colorHist(refrence, histogramsShared_refrence);
    __syncthreads();

    for(int c=0; c < CHANNELS; c++)
    {   
        prefixSum(histogramsShared_target[c],LEVELS, threadIdx.x, blockDim.x);
        prefixSum(histogramsShared_refrence[c], LEVELS, threadIdx.x, blockDim.x);
        __syncthreads();

        for (int i = 0; i < LEVELS; i+=1) {
            for (int j = tid; j < LEVELS; j+=threads) {
                deleta_cdf_row[j] = abs(histogramsShared_target[c][i]-histogramsShared_refrence[c][j]);
            }
            __syncthreads();
            argmin(deleta_cdf_row, LEVELS, threadIdx.x, blockDim.x);
            __syncthreads();

            map_cdf[c][i] = deleta_cdf_row[1];

            __syncthreads();
        }
        __syncthreads();
    }          

    //Preform Map
    performMapping(map_cdf, target, result); 
    __syncthreads(); 
}

class streams_server : public image_processing_server
{
private:
    // TODO define stream server context (memory buffers, streams, etc...)
    int job_id_arr[NUM_STREAMS];
    cudaStream_t streams[NUM_STREAMS];
    queue<int> free_stream;

    uchar *target_single   = nullptr;
    uchar *refrence_single = nullptr;
    uchar *result_single   = nullptr;

public:
    streams_server()
    {
        // TODO initialize context (memory buffers, streams, etc...)
        for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
            job_id_arr[i]=-1;
            free_stream.push(i);
        }
        CUDA_CHECK( cudaMalloc((void**)&(target_single),   img_size) ); 
        CUDA_CHECK( cudaMalloc((void**)&(refrence_single), img_size) ); 
        CUDA_CHECK( cudaMalloc((void**)&(result_single),   img_size) ); 
    }

    ~streams_server() override
    {
        // TODO free resources allocated in constructor
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamDestroy(streams[i]);
        }
    }

    bool enqueue(int job_id, uchar *target, uchar *reference, uchar *result) override
    {
        int num_stream;
        // TODO place memory transfers and kernel invocation in streams if possible.
        if (!free_stream.empty())
        {
            num_stream=free_stream.front();
            free_stream.pop();
            job_id_arr[num_stream]=job_id;
            //printf("job id enqueue: %d",job_id);
            CUDA_CHECK( cudaMemcpy(target_single,   target,   img_size, cudaMemcpyHostToDevice) );
            CUDA_CHECK( cudaMemcpy(refrence_single, reference, img_size, cudaMemcpyHostToDevice) );
            process_image_kernel<<<1,1024,0,streams[num_stream]>>>(target_single,refrence_single,result_single);
            cudaError_t error=cudaGetLastError();
            if (error!=cudaSuccess) 
            {
                fprintf(stderr,"Kernel execution failed:%s\n",cudaGetErrorString(error));
                return 1;
            }
            CUDA_CHECK(cudaMemcpy(result,result_single, img_size, cudaMemcpyDeviceToHost) );
            return true;
        }
        return false;
    }

    bool dequeue(int *job_id) override
    {
        
        for (int i=0;i<NUM_STREAMS;i++)
        {
            cudaError_t status = cudaStreamQuery(streams[i]); // TODO query diffrent stream each iteration
            switch (status) {
            case cudaSuccess:
                    if(job_id_arr[i]==-1)
                        continue;
                    *job_id=job_id_arr[i];
                    job_id_arr[i]=-1;
                    free_stream.push(i);
                return true;
            case cudaErrorNotReady:
                return false;
            default:
                CUDA_CHECK(status);
                return false;
            }
        }
        return false;
    }
};

std::unique_ptr<image_processing_server> create_streams_server()
{
    return std::make_unique<streams_server>();
}

typedef struct Job {
    int job_id;
    uchar* target;
    uchar* reference;
    uchar* result;
}Job;

class Que
{
private:    
    Job que[QUE_MAX] = {0};
    cuda::atomic<int> _head;
    cuda::atomic<int> _tail;

public:
    __device__ __host__ void print() {
        //printf("front:%d, size:%d\n", front_of_que%QUE_MAX, size);
        printf("[%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d]\n", que[0].job_id,que[1].job_id,que[2].job_id,que[3].job_id,que[4].job_id,
                                    que[5].job_id,que[6].job_id,que[7].job_id,que[8].job_id,que[9].job_id,
                                    que[10].job_id,que[11].job_id,que[12].job_id,que[13].job_id,que[14].job_id,
                                     que[15].job_id);
    }

    __device__ __host__  bool dequeue(Job* job) {
        int head = _head.load(cuda::memory_order_relaxed);
        if(_tail.load(cuda::memory_order_acquire) == head) {
            _head.store(head, cuda::memory_order_release);
            return false;
        }
        *job = que[head % QUE_MAX];
        que[head % QUE_MAX] = {-1, NULL, NULL, NULL};
        _head.store(head + 1, cuda::memory_order_release);
        return true;
    }

    __device__ __host__ bool enqueue(Job& job) {
        int tail = _tail.load(cuda::memory_order_relaxed);
        if(tail - _head.load(cuda::memory_order_acquire) == QUE_MAX) {
            _tail.store(tail, cuda::memory_order_release);
            return false;
        }
        que[tail % QUE_MAX] = job;
        _tail.store(tail + 1, cuda::memory_order_release);
        return true;
    }

    __device__ __host__ bool enqueue(int job_id, uchar* target, uchar* reference, uchar* result) {
        int tail = _tail.load(cuda::memory_order_relaxed);
        if(tail - _head.load(cuda::memory_order_acquire) == QUE_MAX){
            _tail.store(tail, cuda::memory_order_release);
            return false;
        }
        assert(target);
        assert(reference);
        assert(result);
        que[tail % QUE_MAX] = {job_id, target, reference, result};;
        _tail.store(tail + 1, cuda::memory_order_release);
        return true;
    }
};

__global__ void kernel(Que* que_host_to_gpu, Que* que_gpu_to_host, bool* running)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ int deleta_cdf_row[LEVELS];
    __shared__ int map_cdf[CHANNELS][LEVELS];
    __shared__ int histogramsShared_target[CHANNELS][LEVELS];
    __shared__ int histogramsShared_refrence[CHANNELS][LEVELS];
    __shared__ bool new_job;
    __shared__ Job job;

    if(tid == 0) {
        new_job = false;
        job = {-1, NULL,NULL,NULL};
    }
    while(*running) {
        if(tid == 0 && que_host_to_gpu[bid].dequeue(&job)) {
            //printf("JOB ID: %d OUT host_to_gpu\n", job.job_id);
            new_job = true;
        }
        __syncthreads();
        if(new_job) {
            if (tid == 0) {
                assert(job.target);
                assert(job.reference);
                assert(job.result);
            }
            process_image(job.target, job.reference, job.result, deleta_cdf_row, map_cdf, histogramsShared_target, histogramsShared_refrence);
            if (tid == 0) {
                que_gpu_to_host[bid].enqueue(job);
                new_job = false;
            }
        }
        __syncthreads();
    }

}


// TODO implement a function for calculating the threadblocks count
__host__
int calculate_blocks_num(int threads) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int number_of_sm = prop.multiProcessorCount;
    int threads_per_block = 256;
    
    int min_blocks = threads/threads_per_block;

    int registers_per_thread = 32;
    int registers_per_block = registers_per_thread * threads_per_block; 
    int regs_per_sm  = prop.regsPerMultiprocessor;
    int total_regs = regs_per_sm * number_of_sm;

    min_blocks = min(min_blocks, total_regs/registers_per_block);

    int threads_per_sm = prop.maxThreadsPerMultiProcessor;
    int total_threads = number_of_sm * threads_per_sm;

    min_blocks = min(min_blocks, total_threads/threads_per_block);

    int shared_mem_per_sm = prop.sharedMemPerMultiprocessor;
    int total_shared_mem = number_of_sm * shared_mem_per_sm;
    int shared_mem_per_block = 10280;

    min_blocks = min(min_blocks, total_shared_mem/shared_mem_per_block);
    return min_blocks;
}


class queue_server : public image_processing_server
{
private:
    char* pinned_host_buffer;
    Que* que_host_to_gpu;
    Que* que_gpu_to_host;
    bool* running;
public:
    // Job& operator[](int index) {
    //     return que[index % que_max];
    // }


    queue_server(int threads) {
        int blocks = calculate_blocks_num(threads);
        cudaMallocHost(&pinned_host_buffer, 2 * blocks * sizeof(Que));
        cudaMallocHost(&running, sizeof(bool));
        // Use placement new operator to construct our class on the pinned buffer
        for(int i = 0; i < 2 * blocks; i++) {
            new (pinned_host_buffer + i * sizeof(Que)) Que();
        }
        que_host_to_gpu = (Que*) pinned_host_buffer; 
        que_gpu_to_host = (Que*) (pinned_host_buffer + blocks * sizeof(Que));
        running = new (running) bool(true);

        kernel<<<blocks, 256>>>(que_host_to_gpu, que_gpu_to_host, running);
    }

    ~queue_server() override {
        *running = false;
        cudaDeviceSynchronize();
        que_host_to_gpu->~Que();
        que_gpu_to_host->~Que();
        auto err = cudaFreeHost(pinned_host_buffer);
        assert(err == cudaSuccess);
        err = cudaFreeHost(running);
        assert(err == cudaSuccess);
    }

    bool enqueue(int job_id, uchar *target, uchar *reference, uchar *result) override
    {
        assert(job_id != -1);
        if(que_host_to_gpu->enqueue(job_id, target, reference, result)){
            //printf("JOB ID: %d IN host_to_gpu\n", job_id);
            return true;
        }
        return false;
    }

    bool dequeue(int *job_id) override {
        auto success = false;
        Job job = {-1, NULL, NULL, NULL};
        if(que_gpu_to_host->dequeue(&job)) {
            success = true;
            *job_id = job.job_id;
            //printf("JOB ID: %d OUT gpu_to_host\n", job.job_id);
            assert(job.reference && job.result && job.target);
        }
        return success;
    }
};

std::unique_ptr<image_processing_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}