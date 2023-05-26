#include "ex2.h"
#include <cuda/atomic>
using namespace std;

#define NUM_STREAMS 64

// Our functions from ex 1
const int img_size = SIZE * SIZE * CHANNELS;
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

__device__ void colorHist(uchar img[][CHANNELS], int histograms[][LEVELS]) {
    const int pic_size = SIZE * SIZE;
    const int tid = threadIdx.x;
    const int threads = blockDim.x;

    for (int i = tid; i < 3*pic_size; i+=threads) {
        const int color = i%3;
        const int pixel = i/3;
        assert(pixel < pic_size);
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
// Our functions from ex 1 end

__device__
void process_image(uchar *targets, uchar *references, uchar *results) {
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
    auto refrence = (uchar(*)[CHANNELS]) &references[bid * img_size];
    auto result   = (uchar(*)[CHANNELS]) &results[  bid * img_size];

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

__global__
void process_image_kernel(uchar *targets, uchar *references, uchar *results){
    process_image(targets, references, results);
}

class streams_server : public image_processing_server
{
private:
    // TODO define stream server context (memory buffers, streams, etc...)
    int current_stream;
    cudaStream_t streams[NUM_STREAMS];


public:
    streams_server()
    {
        // TODO initialize context (memory buffers, streams, etc...)
        current_stream=0;
        

        for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);

        }
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
        // TODO place memory transfers and kernel invocation in streams if possible.
        return false;
    }

    bool dequeue(int *job_id) override
    {
        return false;

        // TODO query (don't block) streams for any completed requests.
        //for ()
        //{
            cudaError_t status = cudaStreamQuery(0); // TODO query diffrent stream each iteration
            switch (status) {
            case cudaSuccess:
                // TODO return the img_id of the request that was completed.
                //*img_id = ...
                return true;
            case cudaErrorNotReady:
                return false;
            default:
                CUDA_CHECK(status);
                return false;
            }
        //}
    }
};

std::unique_ptr<image_processing_server> create_streams_server()
{
    return std::make_unique<streams_server>();
}

// TODO implement a SPSC queue
// TODO implement the persistent kernel
// TODO implement a function for calculating the threadblocks count

typedef struct Job {
    int job_id;
    uchar* target;
    uchar* reference;
    uchar* result;
}Job;

class queue_server : public image_processing_server
{
private:
    Job* que;
    const int que_max = 16;
    int front_of_que = 0;
    int size = 0;
    bool done = false;

public:
    Job& operator[](int index) {
        return que[index % que_max];
    }


    queue_server(int threads) {
        CUDA_CHECK( cudaMallocHost((void**)&que, que_max * sizeof(Job)));

        // TODO initialize host state
        // TODO launch GPU persistent kernel with given number of threads, and calculated number of threadblocks
    }

    ~queue_server() override {
        CUDA_CHECK( cudaFree((void*) que) );
    }

    bool enqueue(int job_id, uchar *target, uchar *reference, uchar *result) override
    {
        //lock
        bool success = false;
        if(size < que_max) {
            que[front_of_que + size] = {job_id, target, reference, result};
            size += 1;
            success = true;
        }
        //unlock
        return success;
    }

    bool dequeue(int *job_id) override
    {
        //lock
        bool success = false;
        if(done) {
            success = true;
            *job_id = que[front_of_que].job_id;
            size -= 1;
            front_of_que += 1;
        }
        //unlock
        return success;
    }
};

std::unique_ptr<image_processing_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}
