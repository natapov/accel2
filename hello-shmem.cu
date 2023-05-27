#include <iostream>
#include <cuda/atomic>
#include "ex2.h"
#define QUE_MAX 16

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
        cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
        return;
        //printf("front:%d, size:%d\n", front_of_que%QUE_MAX, size);
        printf("[%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d]\n", que[0].job_id,que[1].job_id,que[2].job_id,que[3].job_id,que[4].job_id,
                                    que[5].job_id,que[6].job_id,que[7].job_id,que[8].job_id,que[9].job_id,
                                    que[10].job_id,que[11].job_id,que[12].job_id,que[13].job_id,que[14].job_id,
                                     que[15].job_id);
    }

    __device__ __host__ void dequeue(Job* job) {
        int head = _head.load(cuda::memory_order_relaxed);
        while(_tail.load(cuda::memory_order_acquire) == head);
        *job = que[head % QUE_MAX];
        que[head % QUE_MAX] = {-1, NULL, NULL, NULL};
        _head.store(head + 1, cuda::memory_order_release);
        return;
    }

    __device__ __host__ void enqueue(Job& job) {
        int tail = _tail.load(cuda::memory_order_relaxed);
        while(tail - _head.load(cuda::memory_order_acquire) == QUE_MAX);
        que[tail % QUE_MAX] = job;
        _tail.store(tail + 1, cuda::memory_order_release);
        return;
    }

    __device__ __host__ void enqueue(int job_id, uchar* target, uchar* reference, uchar* result) {
        int tail = _tail.load(cuda::memory_order_relaxed);
        while(tail - _head.load(cuda::memory_order_acquire) == QUE_MAX);
        que[tail % QUE_MAX] = {job_id, target, reference, result};;
        _tail.store(tail + 1, cuda::memory_order_release);
        return;
    }
};


__global__ void kernel(Que* shmem_input, Que* shmem_output)
{
    Job c;
    do {
        shmem_input->dequeue(&c);
        shmem_output->enqueue(c);
    } while (c.job_id);
}

int main(int argc, char* argv[]) {
    char *pinned_host_buffer;

    // Allocate pinned host buffer for two shared_memory instances
    cudaMallocHost(&pinned_host_buffer, 2 * sizeof(Que));
    // Use placement new operator to construct our class on the pinned buffer
    Que *que_host_to_gpu = new (pinned_host_buffer) Que();
    Que *que_gpu_to_host = new (pinned_host_buffer + sizeof(Que)) Que();

    bool verbose = true;
    std::string message_to_gpu = "Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory! Hello shared memory!";
    size_t msg_len = message_to_gpu.length();

    if (argc > 1) {
        msg_len = atoi(argv[1]);
        message_to_gpu.resize(msg_len);
        for (size_t i = 0; i < msg_len; ++i)
            message_to_gpu[i] = i & 0xff | 1;
        verbose = false;
    }

    auto message_from_gpu = std::string(msg_len, '\0');

    // Invoke kernel asynchronously
    kernel<<<1, 1>>>(que_host_to_gpu, que_gpu_to_host);

    std::cout << "Writing message to GPU:" << std::endl;
    size_t out = 0;
    for (size_t in = 0; out < msg_len;) {
        char c = message_to_gpu[in];
        Job j = {(int)c, NULL, NULL, NULL};
        que_gpu_to_host->enqueue(j);
        in++;

        que_gpu_to_host->dequeue(&j);
        message_from_gpu[out]= (char) j.job_id;
        std::cout << message_from_gpu[out] << std::flush;
    }
    Job n = {0, NULL, NULL, NULL};

    que_host_to_gpu->enqueue(n);

    if (verbose)
        std::cout << "\nresult:\n" << message_from_gpu << std::endl;
    std::cout << "\n" << "Waiting for kernel to complete." << std::endl;

    cudaError_t err = cudaDeviceSynchronize();
    std::cout << cudaGetErrorString(err) << std::endl;

    if (message_from_gpu != message_to_gpu) {
        std::cout << "Error: got different string from GPU." << std::endl;
    }

    // Destroy queues and release memory
    que_host_to_gpu->~Que();
    que_gpu_to_host->~Que();
    err = cudaFreeHost(pinned_host_buffer);
    assert(err == cudaSuccess);

    cudaDeviceReset();

    return 0;
}



