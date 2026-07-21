#define cimg_display 0
#define cimg_use_png
#include "CImg.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <dirent.h>
#include <sys/types.h>

using namespace cimg_library;
using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "Error de CUDA en " << __FILE__ << ":" << __LINE__ \
                      << " código=" << err << " \"" << cudaGetErrorString(err) << "\"\n"; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define TILE_SIZE 16

static double elapsedMs(Clock::time_point t0, Clock::time_point t1){
    return std::chrono::duration_cast<Ms>(t1 - t0).count();
}

// 1: calcular vector promedio
__global__ void kernel_calcular_promedio(const float* d_dataset, float* d_mean, int m, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        float sum = 0.0f;
        for (int k = 0; k < m; ++k) {
            sum += d_dataset[(size_t)k * n + j];
        }
        d_mean[j] = sum / (float)m;
    }
}

// 2: matriz centrada
__global__ void kernel_centrar_datos(float* d_dataset, const float* d_mean, int m, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    int k = blockIdx.y * blockDim.y + threadIdx.y; 

    if (j < n && k < m) {
        d_dataset[(size_t)k * n + j] -= d_mean[j];
    }
}

// 3: matriz de covarianza
__global__ void kernel_covarianza_tiling(const float* d_centered, float* d_cov, int m, int n) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE]; 
    __shared__ float s_B[TILE_SIZE][TILE_SIZE]; 

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty; 
    int col = bx * TILE_SIZE + tx; 

    float pvalue = 0.0f;
    int num_tiles = (m + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        int k_A = t * TILE_SIZE + tx;
        if (row < n && k_A < m) {
            s_A[ty][tx] = d_centered[(size_t)k_A * n + row];
        } else {
            s_A[ty][tx] = 0.0f;
        }

        int k_B = t * TILE_SIZE + ty;
        if (k_B < m && col < n) {
            s_B[ty][tx] = d_centered[(size_t)k_B * n + col];
        } else {
            s_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            pvalue += s_A[ty][i] * s_B[i][tx];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        d_cov[(size_t)row * n + col] = pvalue / (float)m;
    }
}

int main(int argc, char* argv[])
{

    const char* dataset_path = (argc > 1) ? argv[1] : "DIV2K_valid_LR_bicubic/X4";
    int target_w   = (argc > 2) ? atoi(argv[2]) : 64;
    int target_h   = (argc > 3) ? atoi(argv[3]) : 64;
    int num_images = 100; 
    int channels   = 3;
    
    printf("Experimento 1 CUDA\n");
    printf("Explorando directorio %s ...\n", dataset_path);
    
    auto t0 = Clock::now();

    std::vector<std::string> image_paths;
    DIR *dir = opendir(dataset_path);
    if (dir == NULL) {
        fprintf(stderr, "Error: No se pudo abrir la carpeta '%s'.\n", dataset_path);
        return EXIT_FAILURE;
    }

    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        std::string filename = ent->d_name;
        if (filename.length() >= 4) {
            std::string ext = filename.substr(filename.length() - 4);
            if (ext == ".png") { 
                std::string full_path = std::string(dataset_path) + "/" + filename;
                image_paths.push_back(full_path);
            }
        }
    }
    closedir(dir);

    std::sort(image_paths.begin(), image_paths.end());

    if (image_paths.empty()) {
        fprintf(stderr, "Error: No se encontraron imagenes en '%s'.\n", dataset_path);
        return EXIT_FAILURE;
    }

    num_images = std::min(num_images, (int)image_paths.size());
    int n = target_w * target_h * channels;
    
    size_t dataset_bytes = (size_t)num_images * n * sizeof(float);
    size_t mean_bytes    = (size_t)n * sizeof(float);
    size_t C_bytes       = (size_t)n * n * sizeof(float);

    printf("Imágenes a procesar: %d\n", num_images);
    printf("Resolución   : %d × %d × %d canales\n", target_w, target_h, channels);
    printf("Vector n     : %d\n\n", n);

    float* h_dataset = new float[(size_t)num_images * n];

    for (int k = 0; k < num_images; k++) {
        CImg<unsigned char> img(image_paths[k].c_str());
        img.resize(target_w, target_h, 1, channels);

        float* row_ptr = h_dataset + (size_t)k * n;
        int area = target_w * target_h;
        for (int c = 0; c < channels; c++) {
            for (int y = 0; y < target_h; y++) {
                for (int x = 0; x < target_w; x++) {
                    row_ptr[c * area + y * target_w + x] = (float)img(x, y, 0, c) / 255.0f;
                }
            }
        }
        
        if ((k + 1) % 20 == 0) {
            printf("  %d/%d imágenes cargadas...\n", k + 1, num_images);
        }
    }
    
    auto t1 = Clock::now();
    double t_load = elapsedMs(t0, t1);
    printf("Dataset listo (%.2f MB)  →  %.3f ms\n\n", dataset_bytes / 1e6, t_load);

    // alojamiento en GPU y Transferencia
    float* h_C = new float[(size_t)n * n];
    float *d_dataset, *d_mean, *d_C;

    CUDA_CHECK(cudaMalloc((void**)&d_dataset, dataset_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_mean, mean_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, C_bytes));

    cudaEvent_t start_h2d, stop_h2d, stop_kernels, stop_d2h;
    cudaEventCreate(&start_h2d); cudaEventCreate(&stop_h2d);
    cudaEventCreate(&stop_kernels); cudaEventCreate(&stop_d2h);

    CUDA_CHECK(cudaEventRecord(start_h2d));
    CUDA_CHECK(cudaMemcpy(d_dataset, h_dataset, dataset_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop_h2d));

    // ejecución de kernels
    int threadsPerBlock1D = 256;
    int blocksPerGrid1D = (n + threadsPerBlock1D - 1) / threadsPerBlock1D;
    
    dim3 threads2D(TILE_SIZE, TILE_SIZE);
    dim3 blocks_centrado((n + TILE_SIZE - 1) / TILE_SIZE, (num_images + TILE_SIZE - 1) / TILE_SIZE);
    dim3 blocks_cov((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    printf("Ejecutando Kernels en GPU (Stream 0)...\n");
    
    kernel_calcular_promedio<<<blocksPerGrid1D, threadsPerBlock1D>>>(d_dataset, d_mean, num_images, n);
    kernel_centrar_datos<<<blocks_centrado, threads2D>>>(d_dataset, d_mean, num_images, n);
    kernel_covarianza_tiling<<<blocks_cov, threads2D>>>(d_dataset, d_C, num_images, n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_kernels));

    // transferencia device a host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, C_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop_d2h));
    CUDA_CHECK(cudaEventSynchronize(stop_d2h));

    float ms_h2d = 0, ms_kernels = 0, ms_d2h = 0;
    cudaEventElapsedTime(&ms_h2d, start_h2d, stop_h2d);
    cudaEventElapsedTime(&ms_kernels, stop_h2d, stop_kernels);
    cudaEventElapsedTime(&ms_d2h, stop_kernels, stop_d2h);

    double t_total = ms_h2d + ms_kernels + ms_d2h;

    printf("\nResumen:\n");
    printf("Carga imágenes (Host)  : %8.3f ms\n", t_load);
    printf("Copia H2D (Host->GPU)  : %8.3f ms\n", ms_h2d);
    printf("Cómputo Kernels (Neto) : %8.3f ms\n", ms_kernels);
    printf("Copia D2H (GPU->Host)  : %8.3f ms\n", ms_d2h);
    printf("─────────────────────────────────────────────\n");
    printf("TOTAL GPU (Copia+Comp) : %8.3f ms\n", t_total);
    printf("═════════════════════════════════════════════\n");

    // ── Verificación ──────────────────────────────
    printf("\nVerificación (diagonal de C debe ser > 0):\n");
    printf("  C[0][0]         = %.6f\n", h_C[0]);
    printf("  C[n/2][n/2]     = %.6f\n", h_C[(size_t)(n/2)*n + (n/2)]);
    printf("  C[n-1][n-1]     = %.6f\n", h_C[(size_t)(n-1)*n + (n-1)]);

    int ri = n / 3, rj = n * 2 / 3;
    printf("  Simetría C[%d][%d]=%.6f  C[%d][%d]=%.6f\n",
           ri, rj, h_C[(size_t)ri*n+rj],
           rj, ri, h_C[(size_t)rj*n+ri]);

    // ── Liberar memoria ───────────────────────────
    delete[] h_dataset;
    delete[] h_C;
    cudaFree(d_dataset);
    cudaFree(d_mean);
    cudaFree(d_C);
    
    cudaEventDestroy(start_h2d);
    cudaEventDestroy(stop_h2d);
    cudaEventDestroy(stop_kernels);
    cudaEventDestroy(stop_d2h);

    return EXIT_SUCCESS;
}
