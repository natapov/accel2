#include "ex2.h"
#include <cuda/atomic>

__device__
void process_image(uchar *target, uchar *reference, uchar *result) {
    // TODO complete according to hw1
}

__global__
void process_image_kernel(uchar *target, uchar *reference, uchar *result){
    process_image(target, reference, result);
}

class streams_server : public image_processing_server
{
private:
    // TODO define stream server context (memory buffers, streams, etc...)

public:
    streams_server()
    {
        // TODO initialize context (memory buffers, streams, etc...)
    }

    ~streams_server() override
    {
        // TODO free resources allocated in constructor
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
        for ()
        {
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
        }
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
        CUDA_CHECK( cudaMalloc((void**)&que, que_max * sizeof(Job)));

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
