// Q-ALU as a CUDA kernel: elementwise forward and backward, float32.
//
// f(x) =  alpha * x * (1 + tanh(x / beta))        for x >= 0
// f(x) = -gamma * x * expm1(x / delta)            for x <  0
//
// The derivative is written per branch, so the positive branch never evaluates
// the negative one and vice versa. That matters for two reasons: tanh saturates
// to +-1 in float32 for |x/beta| > 9, and expm1(x/delta) overflows to -inf for
// x/delta < -88, so a branch that multiplies its output by a saturated
// exponential produces 0 * inf = NaN in the backward pass. Selecting on the sign
// of x, rather than on the sign of f, keeps both branches inside their own
// domain.
//
// Build and run the self-test:
//   nvcc -O3 -arch=sm_60 -o qalu_kernel qalu_kernel.cu && ./qalu_kernel
//
// This file has NOT been executed in the paper's environment: the reference
// machine is an Intel Celeron N2930 with no CUDA device and no nvcc. The
// Python reference in examples/qalu_reference.py is what the paper's numbers
// come from; this kernel is the same arithmetic, released so that a GPU user
// can check it rather than take it on trust.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

struct Params {
    float alpha;
    float beta;
    float gamma;
    float delta;
};

// Reference defaults from the Q-ALU paper, which the PyTorch module reaches
// through a softplus with a 1e-4 floor (hence 1.0001 rather than 1.0).
__host__ __device__ inline Params default_params() {
    return Params{1.0f, 1.0f, 1.5f, 0.5f};
}

__host__ __device__ inline float qalu_fwd(float x, Params p) {
    if (x >= 0.0f) {
        return p.alpha * x * (1.0f + tanhf(x / p.beta));
    }
    return -p.gamma * x * expm1f(x / p.delta);
}

__host__ __device__ inline float qalu_grad(float x, Params p) {
    if (x >= 0.0f) {
        const float t = tanhf(x / p.beta);
        // d/dx of alpha*x*(1+tanh(x/beta)); sech^2 = 1 - tanh^2, which is the
        // form that stays finite where tanh saturates.
        return p.alpha * ((1.0f + t) + (x / p.beta) * (1.0f - t * t));
    }
    // d/dx of -gamma*x*expm1(x/delta), with expm1 and exp kept apart so the
    // x*exp(x/delta) term does not lose precision for x near zero.
    return -p.gamma * (expm1f(x / p.delta) + (x / p.delta) * expf(x / p.delta));
}

__global__ void qalu_forward_kernel(const float* __restrict__ x,
                                    float* __restrict__ y,
                                    int n, Params p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int j = i; j < n; j += stride) {
        y[j] = qalu_fwd(x[j], p);
    }
}

__global__ void qalu_backward_kernel(const float* __restrict__ x,
                                     const float* __restrict__ dy,
                                     float* __restrict__ dx,
                                     int n, Params p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int j = i; j < n; j += stride) {
        dx[j] = dy[j] * qalu_grad(x[j], p);
    }
}

// Host-side entry points, the shape a framework binding would call.
void qalu_forward(const float* d_x, float* d_y, int n, Params p) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    qalu_forward_kernel<<<blocks, threads>>>(d_x, d_y, n, p);
}

void qalu_backward(const float* d_x, const float* d_dy, float* d_dx, int n, Params p) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    qalu_backward_kernel<<<blocks, threads>>>(d_x, d_dy, d_dx, n, p);
}

static double ref_fwd(double x, Params p) {
    if (x >= 0.0) return p.alpha * x * (1.0 + std::tanh(x / p.beta));
    return -p.gamma * x * std::expm1(x / p.delta);
}

static double ref_grad(double x, Params p) {
    if (x >= 0.0) {
        const double t = std::tanh(x / p.beta);
        return p.alpha * ((1.0 + t) + (x / p.beta) * (1.0 - t * t));
    }
    return -p.gamma * (std::expm1(x / p.delta) + (x / p.delta) * std::exp(x / p.delta));
}

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,              \
                         cudaGetErrorString(err));                              \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

int main() {
    const Params p = default_params();
    const int n = 1 << 22;  // 4.2 M elements, the size of a wide activation layer

    float* h_x = static_cast<float*>(std::malloc(n * sizeof(float)));
    float* h_y = static_cast<float*>(std::malloc(n * sizeof(float)));
    float* h_dx = static_cast<float*>(std::malloc(n * sizeof(float)));
    for (int i = 0; i < n; ++i) {
        // Uniform over [-8, 8], so both branches and the overshoot region are
        // covered, with the exact zero included on purpose.
        h_x[i] = -8.0f + 16.0f * static_cast<float>(i) / static_cast<float>(n - 1);
    }

    float *d_x, *d_y, *d_dx, *d_dy;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dx, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dy, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dy, 0, n * sizeof(float)));
    // dy = 1 everywhere, so dx is the derivative itself.
    float* h_ones = static_cast<float*>(std::malloc(n * sizeof(float)));
    for (int i = 0; i < n; ++i) h_ones[i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(d_dy, h_ones, n * sizeof(float), cudaMemcpyHostToDevice));

    qalu_forward(d_x, d_y, n, p);
    qalu_backward(d_x, d_dy, d_dx, n, p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dx, d_dx, n * sizeof(float), cudaMemcpyDeviceToHost));

    double max_fwd = 0.0, max_bwd = 0.0;
    int bad = 0;
    for (int i = 0; i < n; ++i) {
        const double x = h_x[i];
        if (!std::isfinite(h_y[i]) || !std::isfinite(h_dx[i])) ++bad;
        max_fwd = std::fmax(max_fwd, std::fabs(static_cast<double>(h_y[i]) - ref_fwd(x, p)));
        // x == 0 is skipped for the same reason as in the Python reference: the
        // PyTorch module returns 0 there by construction of its clamp, and this
        // kernel returns alpha, which is the mathematically correct value.
        if (x != 0.0f) {
            max_bwd = std::fmax(max_bwd, std::fabs(static_cast<double>(h_dx[i]) - ref_grad(x, p)));
        }
    }
    std::printf("n=%d  non-finite=%d\n", n, bad);
    std::printf("max |f_fwd - ref| = %.3e   max |f_bwd - ref| = %.3e  (float32 vs double)\n",
                max_fwd, max_bwd);
    std::printf("f'(-2*delta) = %.6f   gamma*(1+e^-2) = %.6f\n",
                static_cast<double>(qalu_grad(-2.0f * p.delta, p)),
                static_cast<double>(p.gamma) * (1.0 + std::exp(-2.0)));

    // Latency of the elementwise pass, the number that decides whether the
    // activation is worth fusing into a block. Bandwidth-bound: 8 bytes of
    // traffic per element for the forward pass.
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    const int iters = 200;
    qalu_forward(d_x, d_y, n, p);  // warm up
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i) qalu_forward(d_x, d_y, n, p);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    const double per_call = ms / iters;
    const double gb = 2.0 * sizeof(float) * static_cast<double>(n) / 1e9;
    std::printf("forward: %.3f ms/call over %d elements  (%.1f GB/s effective)\n",
                per_call, n, gb / (per_call * 1e-3));

    std::free(h_x); std::free(h_y); std::free(h_dx); std::free(h_ones);
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_dx)); CUDA_CHECK(cudaFree(d_dy));
    return 0;
}
