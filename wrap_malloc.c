#include <numa.h>
#include <stdio.h>
#include <stdlib.h>
#include <numaif.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>

#define BLOCK_SIZE (8 * 1024 * 1024) // 512MB 单位块大小
#define PAGE_SIZE (4 * 1024)           // 系统页大小，一般为 4KB

int my_NUMA_NODE_COUNT = 4;               // 动态设置 NUMA 节点数量
void* __real_malloc(size_t size);

void* __wrap_malloc(size_t size) {
    // 检查 libnuma 初始化是否成功
    if (numa_available() < 0) {
        return __real_malloc(size); // 如果 NUMA 不可用，使用原始的 malloc
    }

    // 动态获取 NUMA 节点数量
    int NUMA_NODE_COUNT = numa_max_node() + 2;

    // 计算需要的总块数，大小对齐到页大小
    size_t num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t total_size = num_blocks * BLOCK_SIZE;

    if (total_size % PAGE_SIZE != 0) {
        fprintf(stderr, "Total size is not a multiple of the page size.\n");
        return NULL;
    }

    // 使用 mmap 分配虚拟地址空间
    void* result = mmap(NULL, total_size, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (result == MAP_FAILED) {
        fprintf(stderr, "mmap failed: %s\n", strerror(errno));
        return NULL;
    }

    // 为每个 512MB 块绑定到不同的 NUMA 节点
    unsigned long nodemask[NUMA_NODE_COUNT / (8 * sizeof(unsigned long)) + 1];
    memset(nodemask, 0, sizeof(nodemask)); // 初始化 nodemask 为 0

    for (size_t i = 0; i < num_blocks; i++) {
        // 当前块的地址
        void* block_address = (char*)result + i * BLOCK_SIZE;

        // 当前 NUMA 节点编号
        int current_numa_node = i % my_NUMA_NODE_COUNT;

        // 清空 nodemask
        memset(nodemask, 0, sizeof(nodemask));
        // 设置内存绑定策略为当前 NUMA 节点
        nodemask[current_numa_node / (8 * sizeof(unsigned long))] = 1UL << (current_numa_node % (8 * sizeof(unsigned long)));

        // 打印调试信息
        //fprintf(stderr, "mbind parameters: block_address=%p, size=%zu, nodemask=%lu, node_count=%d\n",
        //        block_address, BLOCK_SIZE, nodemask[0], NUMA_NODE_COUNT);

        int status = mbind(block_address, BLOCK_SIZE, MPOL_BIND, nodemask, NUMA_NODE_COUNT, 0);
        if (status != 0) {
            fprintf(stderr, "mbind failed on node %d: %s\n", current_numa_node, strerror(errno));
            munmap(result, total_size); // 解除映射
            return NULL;
        }

        // 打印调试信息，显示分配的 NUMA 节点和内存块地址
        //fprintf(stderr, "Allocated %zu bytes on NUMA node %d, address: %p\n", BLOCK_SIZE, current_numa_node, block_address);
    }

    return result;
}
