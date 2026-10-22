// CSS314 Task 6: Synchronization Cost & Data Loss
// Build on macOS: clang -O2 -std=c11 -pthread task6.c -o task6
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define THREADS 4
#define ITERATIONS 2000000ULL
#define EXPECTED (THREADS * ITERATIONS)

static atomic_ullong shared_counter;
static unsigned long long protected_counter;
static pthread_mutex_t counter_lock = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    int locked;
} worker_args;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static void *worker(void *arg) {
    const worker_args *settings = (const worker_args *)arg;
    for (unsigned long long i = 0; i < ITERATIONS; ++i) {
        if (settings->locked) {
            pthread_mutex_lock(&counter_lock);
            ++protected_counter;
            pthread_mutex_unlock(&counter_lock);
        } else {
            // Intentionally broken read-modify-write sequence.
            // Atomic load/store avoids undefined C data-race behavior,
            // but the pair is NOT an atomic increment. Updates get lost.
            unsigned long long old = atomic_load_explicit(&shared_counter, memory_order_relaxed);
            atomic_store_explicit(&shared_counter, old + 1, memory_order_relaxed);
        }
    }
    return NULL;
}

static double run_case(int locked) {
    pthread_t threads[THREADS];
    worker_args args = {.locked = locked};
    double start = now_ms();
    for (int i = 0; i < THREADS; ++i) {
        if (pthread_create(&threads[i], NULL, worker, &args) != 0) {
            fprintf(stderr, "Error creating thread %d\n", i);
            exit(EXIT_FAILURE);
        }
    }
    for (int i = 0; i < THREADS; ++i) {
        if (pthread_join(threads[i], NULL) != 0) {
            fprintf(stderr, "Error joining thread %d\n", i);
            exit(EXIT_FAILURE);
        }
    }
    return now_ms() - start;
}

int main(void) {
    atomic_init(&shared_counter, 0);
    protected_counter = 0;

    double unsafe_ms = run_case(0);
    unsigned long long unsafe_result = atomic_load_explicit(&shared_counter, memory_order_relaxed);
    double lost_pct = 100.0 * (double)(EXPECTED - unsafe_result) / (double)EXPECTED;

    double locked_ms = run_case(1);
    double penalty_ms = locked_ms - unsafe_ms;

    printf("Student ID: 230103342\n");
    printf("Threads: %d\n", THREADS);
    printf("Increments per thread: %llu\n", (unsigned long long)ITERATIONS);
    printf("Expected final value: %llu\n\n", (unsigned long long)EXPECTED);
    printf("Unsafe read-modify-write (no lock):\n");
    printf("  Actual final value: %llu\n", unsafe_result);
    printf("  Updates lost: %llu\n", (unsigned long long)EXPECTED - unsafe_result);
    printf("  Percentage lost: %.4f%%\n", lost_pct);
    printf("  Wall-clock time: %.3f ms\n\n", unsafe_ms);
    printf("Correct version (pthread mutex):\n");
    printf("  Actual final value: %llu\n", protected_counter);
    printf("  Wall-clock time: %.3f ms\n\n", locked_ms);
    printf("Synchronization time penalty: %+.3f ms\n", penalty_ms);
    printf("Synchronization time ratio (locked / unsafe): %.3fx\n", locked_ms / unsafe_ms);
    printf("Validation: %s\n", protected_counter == EXPECTED ? "PASS" : "FAIL");
    pthread_mutex_destroy(&counter_lock);
    return protected_counter == EXPECTED ? EXIT_SUCCESS : EXIT_FAILURE;
}
