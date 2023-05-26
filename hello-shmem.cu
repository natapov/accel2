#include <iostream>
#include <cuda/atomic>
#include "ex2.h"

typedef struct Job {
    char data;
    // int job_id;
    // uchar* target;
    // uchar* reference;
    // uchar* result;
}Job;

class Que
{
private:
    // On each use of the flag, we switch the meaning of its values (true/false -> data is ready/not-ready)

    cuda::atomic<bool> lock;
    
    static const int que_max = 16;
    Job que[que_max];
    int front_of_que = 0;
    int size = 0;

public:
    Que() : lock(false){
    }

    __device__ __host__ bool dequeue(char* data)
    {
        bool success = false;
        while (lock.exchange(true ,cuda::memory_order_relaxed) == false);
        cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
        if(size) {
            *data = que[front_of_que%que_max].data;
            size -= 1;
            front_of_que += 1;
            success = true;
        }
        lock.store(false, cuda::memory_order_release);
        return success;
    }

    __device__ __host__ bool enqueue(char value)
    {
        while (lock.exchange(true ,cuda::memory_order_relaxed) == false);
        cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);

        bool success = false;
        if(size < que_max) {
            que[(front_of_que + size)%que_max] = {value};
            // que[front_of_que + size] = {job_id, target, reference, result};

            size += 1;
            success = true;
        }
        lock.store(false, cuda::memory_order_release);
        return success;
    }
};

__global__ void kernel(Que* shmem_input, Que* shmem_output)
{
    char c;
    do {
        auto result = shmem_input->dequeue(&c);
        if(result) {
            shmem_output->enqueue(c);
        }
    } while (c);
}

int main(int argc, char* argv[]) {
    char *pinned_host_buffer;

    // Allocate pinned host buffer for two shared_memory instances
    cudaMallocHost(&pinned_host_buffer, 2 * sizeof(Que));
    // Use placement new operator to construct our class on the pinned buffer
    Que *que_host_to_gpu = new (pinned_host_buffer) Que();
    Que *que_gpu_to_host = new (pinned_host_buffer + sizeof(Que)) Que();

    bool verbose = true;
    std::string message_to_gpu = "Hello shared memory!";
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

    for (size_t i = 0; i < msg_len; ++i) {
        char c = message_to_gpu[i];
        while(!que_host_to_gpu->enqueue(c));
        while(!que_gpu_to_host->dequeue(&message_from_gpu[i]));
        if (verbose)
            std::cout << c << std::flush;
    }
    que_host_to_gpu->enqueue(0);

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



