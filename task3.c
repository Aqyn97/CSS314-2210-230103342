// CSS314 Task 3: False Sharing Profile
// Build on macOS: clang -O2 -std=c11 -pthread task3.c -o task3
#define _POSIX_C_SOURCE 200809L
#ifdef __APPLE__
#define _DARWIN_C_SOURCE 1
#endif
#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#define STUDENT_ID "230103342"
#define ITERATIONS UINT64_C(50000000)
#define CACHE_LINE_BYTES 64

typedef struct {
    volatile uint64_t value;
    unsigned char padding[CACHE_LINE_BYTES - sizeof(uint64_t)];
} PaddedCounter;
_Static_assert(sizeof(PaddedCounter) == CACHE_LINE_BYTES, "Counter must occupy one cache line");

typedef struct {
    int index;
    int padded;
    volatile uint64_t *shared;
    PaddedCounter *separated;
    atomic_int *start;
} WorkerArgs;

static int physical_cores(void) {
#ifdef __APPLE__
    int cores = 0;
    size_t size = sizeof(cores);
    if (sysctlbyname("hw.physicalcpu", &cores, &size, NULL, 0) == 0 && cores > 0)
        return cores;
    return 1; /* macOS fallback if hw.physicalcpu is unavailable */
#else
    long logical = sysconf(_SC_NPROCESSORS_ONLN);
    return logical > 0 ? (int)logical : 1;
#endif
}

static double now_ms(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t) != 0) { perror("clock_gettime"); exit(EXIT_FAILURE); }
    return (double)t.tv_sec * 1000.0 + (double)t.tv_nsec / 1000000.0;
}

static void *worker(void *data) {
    WorkerArgs *args = data;
    while (atomic_load_explicit(args->start, memory_order_acquire) == 0) { }
    if (args->padded) {
        volatile uint64_t *counter = &args->separated[args->index].value;
        for (uint64_t j = 0; j < ITERATIONS; ++j) ++(*counter);
    } else {
        volatile uint64_t *counter = &args->shared[args->index];
        for (uint64_t j = 0; j < ITERATIONS; ++j) ++(*counter);
    }
    return NULL;
}

static double benchmark(int cores, int padded) {
    pthread_t *threads = calloc((size_t)cores, sizeof(*threads));
    WorkerArgs *args = calloc((size_t)cores, sizeof(*args));
    volatile uint64_t *shared = NULL;
    PaddedCounter *separated = NULL;
    atomic_int start = ATOMIC_VAR_INIT(0);
    if (!threads || !args) { fprintf(stderr, "Memory allocation failed\n"); exit(EXIT_FAILURE); }
    if (padded) {
        void *memory = NULL;
        if (posix_memalign(&memory, CACHE_LINE_BYTES, (size_t)cores * sizeof(PaddedCounter))) {
            fprintf(stderr, "Aligned allocation failed\n"); exit(EXIT_FAILURE);
        }
        separated = memory;
        for (int i = 0; i < cores; ++i) separated[i].value = 0;
    } else {
        shared = calloc((size_t)cores, sizeof(*shared));
        if (!shared) { fprintf(stderr, "Memory allocation failed\n"); exit(EXIT_FAILURE); }
    }
    for (int i = 0; i < cores; ++i) {
        args[i] = (WorkerArgs){ i, padded, shared, separated, &start };
        if (pthread_create(&threads[i], NULL, worker, &args[i]) != 0) {
            fprintf(stderr, "Thread creation failed\n"); exit(EXIT_FAILURE);
        }
    }
    double begin = now_ms();
    atomic_store_explicit(&start, 1, memory_order_release);
    for (int i = 0; i < cores; ++i) pthread_join(threads[i], NULL);
    double elapsed = now_ms() - begin;
    for (int i = 0; i < cores; ++i) {
        uint64_t result = padded ? separated[i].value : shared[i];
        if (result != ITERATIONS) {
            fprintf(stderr, "Incorrect result for thread %d: %" PRIu64 "\n", i, result);
            exit(EXIT_FAILURE);
        }
    }
    free((void *)shared);
    free(separated);
    free(args);
    free(threads);
    return elapsed;
}

int main(void) {
    int cores = physical_cores();
    printf("Student ID: %s\n", STUDENT_ID);
    printf("Physical cores used (threads): %d\n", cores);
    printf("Increments per thread: %" PRIu64 "\n", ITERATIONS);
    printf("Counter size: %zu bytes; padded stride: %zu bytes\n\n",
           sizeof(uint64_t), sizeof(PaddedCounter));
    double unpadded = benchmark(cores, 0);
    double padded = benchmark(cores, 1);
    printf("%-24s %14s\n", "Configuration", "Wall Time (ms)");
    printf("-----------------------------------------\n");
    printf("%-24s %14.3f\n", "Unpadded Shared", unpadded);
    printf("%-24s %14.3f\n", "Padded (64-byte)", padded);
    printf("\nExecution ratio (Unpadded / Padded): %.3fx\n", unpadded / padded);
    printf("Validation: every thread completed %" PRIu64 " increments in both runs.\n", ITERATIONS);
    return 0;
}
