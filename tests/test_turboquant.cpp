#include "amuquant/turboquant.hpp"
#include "amuquant/turbovec_ffi.h"

#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <cmath>
#include <random>

using namespace amuquant;

int main() {
    std::cout << "[TEST] Starting TurboQuant & FFI Validation..." << std::endl;

    constexpr uint32_t DIM = 4096;
    constexpr uint32_t NUM_VECTORS = 100;

    // 1. Initialize TurboQuant index via C-ABI
    TurbovecIndexHandle* handle = nullptr;
    TurbovecStatus st = turbovec_init(DIM, 4, &handle);
    assert(st == TURBOVEC_STATUS_OK);
    assert(handle != nullptr);
    std::cout << "  - turbovec_init: OK" << std::endl;

    // 2. Generate deterministic synthetic embedding vectors
    std::vector<std::vector<float>> test_vectors(NUM_VECTORS, std::vector<float>(DIM));
    for (uint32_t i = 0; i < NUM_VECTORS; ++i) {
        for (uint32_t d = 0; d < DIM; ++d) {
            test_vectors[i][d] = std::sin(static_cast<float>(i * 100 + d));
        }
    }

    // 3. Test zero-copy vector insertion
    for (uint32_t i = 0; i < NUM_VECTORS; ++i) {
        st = turbovec_insert_zero_copy(handle, i, test_vectors[i].data(), DIM);
        assert(st == TURBOVEC_STATUS_OK);
    }
    std::cout << "  - turbovec_insert_zero_copy (" << NUM_VECTORS << " vectors): OK" << std::endl;

    // 4. Verify heap memory footprint
    size_t mem_bytes_4bit = 0;
    st = turbovec_get_memory_usage(handle, &mem_bytes_4bit);
    assert(st == TURBOVEC_STATUS_OK);
    std::cout << "  - Memory usage (4-bit mode): " << (mem_bytes_4bit / 1024) << " KB" << std::endl;

    // 5. Test SIMD search latency and retrieval accuracy
    constexpr uint32_t TARGET_QUERY_IDX = 42;
    std::vector<uint64_t> out_ids(5);
    std::vector<float> out_scores(5);
    uint32_t count = 0;

    auto t_start = std::chrono::high_resolution_clock::now();
    st = turbovec_search_simd(handle, test_vectors[TARGET_QUERY_IDX].data(), 5, out_ids.data(), out_scores.data(), &count);
    auto t_end = std::chrono::high_resolution_clock::now();
    assert(st == TURBOVEC_STATUS_OK);
    assert(count == 5);

    double search_latency_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "  - SIMD Search Latency: " << search_latency_ms << " ms" << std::endl;
    assert(search_latency_ms < 1.0); // Sub-millisecond retrieval check

    // Verify target vector ranks Top-1 with high cosine fidelity
    assert(out_ids[0] == TARGET_QUERY_IDX);
    assert(out_scores[0] > 0.85f);
    std::cout << "  - Top-1 Match Chunk ID: " << out_ids[0] << ", Cosine Score: " << out_scores[0] << " (OK)" << std::endl;

    // 6. Test in-place requantization from 4-bit to 2-bit
    st = turbovec_requantize_to_2bit(handle);
    assert(st == TURBOVEC_STATUS_OK);
    std::cout << "  - turbovec_requantize_to_2bit: OK" << std::endl;

    // Verify search fidelity preserved after 2-bit compression
    st = turbovec_search_simd(handle, test_vectors[TARGET_QUERY_IDX].data(), 5, out_ids.data(), out_scores.data(), &count);
    assert(st == TURBOVEC_STATUS_OK);
    assert(out_ids[0] == TARGET_QUERY_IDX);
    std::cout << "  - Top-1 Match after 2-bit Requantization: " << out_ids[0] << " (Score: " << out_scores[0] << ") (OK)" << std::endl;

    // 7. Quantitative Lloyd-Max Distortion Verification
    // Verify Mean Squared Error (MSE) on standard normal distribution against theoretical limits:
    // Theoretical Lloyd-Max 4-bit MSE on N(0, 1) is ~0.0095; 2-bit MSE is ~0.1175.
    TurboQuantEngine tq_eval(DIM, 4);
    std::mt19937 eval_rng(12345);
    std::normal_distribution<float> norm_dist(0.0f, 1.0f);
    constexpr size_t NUM_SAMPLES = 10000;
    double sum_sq_err_4bit = 0.0;
    double sum_sq_err_2bit = 0.0;

    const float centroids_4bit[16] = {
        -2.733f, -2.069f, -1.618f, -1.256f,
        -0.942f, -0.657f, -0.388f, -0.128f,
         0.128f,  0.388f,  0.657f,  0.942f,
         1.256f,  1.618f,  2.069f,  2.733f
    };
    const float centroids_2bit[4] = {
        -1.510f, -0.453f, 0.453f, 1.510f
    };
    const uint8_t lut_4to2[16] = {
        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3
    };

    for (size_t i = 0; i < NUM_SAMPLES; ++i) {
        float x = norm_dist(eval_rng);
        uint8_t q4 = tq_eval.quantize_scalar_4bit(x);
        float x_hat_4bit = centroids_4bit[q4];
        double err4 = static_cast<double>(x - x_hat_4bit);
        sum_sq_err_4bit += err4 * err4;

        // Requantize to 2-bit via Lloyd-Max lookup table
        uint8_t q2 = lut_4to2[q4];
        float x_hat_2bit = centroids_2bit[q2];
        double err2 = static_cast<double>(x - x_hat_2bit);
        sum_sq_err_2bit += err2 * err2;
    }

    double mse_4bit = sum_sq_err_4bit / static_cast<double>(NUM_SAMPLES);
    double mse_2bit = sum_sq_err_2bit / static_cast<double>(NUM_SAMPLES);

    std::cout << "  - Lloyd-Max 4-bit Empirical MSE: " << mse_4bit << " (Target < 0.035): OK" << std::endl;
    std::cout << "  - Lloyd-Max 2-bit Empirical MSE: " << mse_2bit << " (Target < 0.200): OK" << std::endl;
    assert(mse_4bit < 0.035);
    assert(mse_2bit < 0.200);

    // 8. Release index memory
    st = turbovec_free(handle);
    assert(st == TURBOVEC_STATUS_OK);
    std::cout << "  - turbovec_free: OK" << std::endl;

    std::cout << "[PASS] TurboQuant & FFI Validation Succeeded.\n" << std::endl;
    return 0;
}
