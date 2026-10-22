/* CSS314 Task 7: Arithmetic Intensity vs Memory Bottlenecks
 * Build (macOS): clang -O2 -std=gnu11 -pthread task7.c -o task7 -lm
 */
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#define N 30000000UL
#define TRIALS 3

typedef enum { COPY, COMPUTE } Kernel;
typedef struct {
    const double *a;
    double *b;
    size_t start, end;
    Kernel kernel;
} Job;

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1000.0 + (double)t.tv_nsec / 1000000.0;
}

static int core_count(void) {
#ifdef __APPLE__
    int n = 0;
    size_t len = sizeof(n);
    if (sysctlbyname("hw.logicalcpu", &n, &len, NULL, 0) == 0 && n > 0)
        return n;
#endif
    long fallback_cores = sysconf(_SC_NPROCESSORS_ONLN);
    return fallback_cores > 0 ? (int)fallback_cores : 1;
}

static void *worker(void *arg) {
    Job *j = (Job *)arg;
    if (j->kernel == COPY) {
        for (size_t i = j->start; i < j->end; ++i)
            j->b[i] = j->a[i];
    } else {
        for (size_t i = j->start; i < j->end; ++i) {
            double x = j->a[i];
            j->b[i] = sin(x) * cos(x) + tan(x);
        }
    }
    return NULL;
}

static double run(Kernel k, int workers, const double *a, double *b) {
    pthread_t *threads = calloc((size_t)workers, sizeof(*threads));
    Job *jobs = calloc((size_t)workers, sizeof(*jobs));
    if (!threads || !jobs) { perror("allocating worker metadata"); exit(1); }
    double start = now_ms();
    for (int t = 0; t < workers; ++t) {
        jobs[t] = (Job){a, b, N * (size_t)t / (size_t)workers,
                        N * (size_t)(t + 1) / (size_t)workers, k};
        if (pthread_create(&threads[t], NULL, worker, &jobs[t]) != 0) {
            fprintf(stderr, "pthread_create failed\n"); exit(1);
        }
    }
    for (int t = 0; t < workers; ++t) pthread_join(threads[t], NULL);
    double elapsed = now_ms() - start;
    free(jobs);
    free(threads);
    return elapsed;
}

static double median3(double x[TRIALS]) {
    for (int i = 0; i < TRIALS; ++i)
        for (int j = i + 1; j < TRIALS; ++j)
            if (x[j] < x[i]) { double tmp = x[i]; x[i] = x[j]; x[j] = tmp; }
    return x[TRIALS / 2];
}

int main(void) {
    const int max_cores = core_count();
    const int counts[3] = {1, 2, max_cores};
    double *a = malloc(N * sizeof(*a));
    double *b = malloc(N * sizeof(*b));
    if (!a || !b) { fprintf(stderr, "Need roughly 480 MB free RAM.\n"); free(a); free(b); return 1; }
    for (size_t i = 0; i < N; ++i) a[i] = 0.01 + 0.99 * ((double)(i % 10000) / 10000.0);
    memset(b, 0, N * sizeof(*b));

    printf("Student ID: 230103342\n");
    printf("Array size: %lu elements (float64)\n", (unsigned long)N);
    printf("Logical cores detected: %d\n", max_cores);
    printf("Threads tested: 1, 2, %d\n", max_cores);
    printf("Trials per configuration: %d (median shown)\n", TRIALS);
    printf("Timing includes thread creation and joining.\n");
    printf("Input range: [0.01, 1.0), avoiding tangent singularities.\n\n");

    double speedups[2][3] = {{0}};
    for (int k = 0; k < 2; ++k) {
        printf("Kernel %c (%s)\n", k == 0 ? 'A' : 'B',
               k == 0 ? "Memory-bound vector copy" : "Compute-heavy trig expression");
        printf("%8s | %15s | %9s | %11s\n", "Workers", "Wall time (ms)", "Speedup", "Efficiency");
        printf("---------------------------------------------------------\n");
        double baseline = 0;
        for (int j = 0; j < 3; ++j) {
            int workers = counts[j];
            if (workers < 1) workers = 1;
            (void)run((Kernel)k, workers, a, b); /* warm-up */
            double times[TRIALS];
            for (int r = 0; r < TRIALS; ++r)
                times[r] = run((Kernel)k, workers, a, b);
            double ms = median3(times);
            if (j == 0) baseline = ms;
            double speedup = baseline / ms;
            double efficiency = speedup / workers;
            speedups[k][j] = speedup;
            printf("%8d | %15.3f | %8.3fx | %10.2f%%\n",
                   workers, ms, speedup, efficiency * 100.0);
            /* check the result after measurement to keep computation observable */
            size_t ix = N / 2;
            double expected = k == COPY ? a[ix] : sin(a[ix]) * cos(a[ix]) + tan(a[ix]);
            if (fabs(b[ix] - expected) > 1e-12) {
                fprintf(stderr, "Result validation failed\n"); free(a); free(b); return 1;
            }
        }
        puts("");
    }
    int last = 2;
    printf("Scaling diagnosis (based on measured results):\n");
    printf("  Copy speedup at %d workers: %.3fx\n", counts[last], speedups[0][last]);
    printf("  Trig speedup at %d workers: %.3fx\n", counts[last], speedups[1][last]);
    if (speedups[0][last] < speedups[1][last] && speedups[0][last] < 0.8 * counts[last]) {
        puts("  Kernel A (vector copy) stalled relative to Kernel B, consistent with memory bandwidth saturation.");
        puts("  Note: this suggests saturation; bandwidth counters would be needed to confirm the cause.");
    } else if (speedups[0][last] < 0.8 * counts[last]) {
        puts("  Kernel A (vector copy) scaled below ideal, but memory saturation is NOT confirmed by this comparison.");
        puts("  Thread overhead, cache effects, and RAM bandwidth are possible contributors.");
    } else {
        puts("  Kernel A (vector copy) did not show a clear memory-bandwidth scaling stall in this run.");
    }
    printf("Validation: both kernels produced correct sampled results.\n");
    free(a);
    free(b);
    return 0;
}
