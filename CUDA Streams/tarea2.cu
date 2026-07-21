#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define TILE_DIM 16

// Macro para capturar y reportar errores de CUDA
#define checkCudaError(call)                                                 \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA Error en " << __FILE__ << ":" << __LINE__     \
                      << " -> " << cudaGetErrorString(err) << std::endl;     \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

extern void cargarImagenesHost(float* h_dataset, int num_images, int n, int width, int height, int channels);
extern void obtenerDimensionesHost(int& w, int& h, int& c);

// Kernel 1: Acumula la suma local de cada lote para el calculo del promedio
__global__ void kernelSumarLote(const float* d_dataset, float* d_mu_sum, int m_batch, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    if (j < n) {
        float suma_local = 0.0f;
        for (int k = 0; k < m_batch; k++) {
            suma_local += d_dataset[k * n + j];
        }
        atomicAdd(&d_mu_sum[j], suma_local);
    }
}

// Kernel 2: Divide la suma total por 'm' para obtener el vector promedio real
__global__ void kernelFinalizarPromedio(float* d_mu, int m) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    d_mu[j] = d_mu[j] / m;
}

// Kernel 3: Resta el vector promedio global a las imagenes del lote
__global__ void kernelCentrarDatos(const float* d_dataset, const float* d_mu, float* d_centered, int m_batch, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int j = idx % n; 
    int k = idx / n;
    if (k < m_batch && j < n) {
        d_centered[k * n + j] = d_dataset[k * n + j] - d_mu[j];
    }
}

// Kernel 4: Multiplicacion matricial optimizada usando Memoria Compartida (Tiling)
__global__ void kernelAcumularCovarianzaTiled(const float* d_centered, float* d_C, int m_batch, int n) {
    __shared__ float tileA[TILE_DIM][TILE_DIM];
    __shared__ float tileB[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float suma = 0.0f;

    for (int t = 0; t < (m_batch + TILE_DIM - 1) / TILE_DIM; ++t) {
        int kA = t * TILE_DIM + threadIdx.x;
        if (row < n && kA < m_batch) {
            tileA[threadIdx.y][threadIdx.x] = d_centered[kA * n + row];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int kB = t * TILE_DIM + threadIdx.y;
        if (col < n && kB < m_batch) {
            tileB[threadIdx.y][threadIdx.x] = d_centered[kB * n + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k) {
            suma += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        atomicAdd(&d_C[row * n + col], suma);
    }
}

// Kernel 5: Aplica la division final por 'm' a la matriz de covarianza
__global__ void kernelFinalizarCovarianza(float* d_C, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        d_C[idx] = d_C[idx] / m;
    }
}

int main(int argc, char* argv[]) {
    int S = (argc > 1) ? atoi(argv[1]) : 4; 
    int num_images = 100; 
    int width, height, channels;

    obtenerDimensionesHost(width, height, channels);
    
    // Forzamos un tamano maximo para el vector n que tu GPU pueda manejar en covarianza (n x n)
    int n_original = width * height * channels;
    int n = 32768; // Cambia este valor (ej: 1024, 2048, 4096) para ajustar el tamano del problema
    if (n > n_original) n = n_original;

    // Validacion adaptada a la VRAM fisica libre en la GPU
    size_t cov_size = (size_t)n * n * sizeof(float);
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);

    std::cout << "Dimensiones originales: " << width << "x" << height << "x" << channels << " (n=" << n_original << ")" << std::endl;
    std::cout << "Dimensiones acotadas para simulacion segura: n = " << n << std::endl;
    std::cout << "Tamano de matriz C solicitado: " << cov_size / (1024*1024) << " MB" << std::endl;

    if (cov_size > (size_t)(free_mem * 0.85)) { 
        std::cerr << "ALERTA: Supera la memoria disponible." << std::endl;
        return -1;
    }

    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start)); 
    checkCudaError(cudaEventCreate(&stop));

    float *h_dataset, *h_C_result;
    size_t dataset_bytes = (size_t)num_images * n * sizeof(float);
    
    checkCudaError(cudaMallocHost((void**)&h_dataset, dataset_bytes)); 
    checkCudaError(cudaMallocHost((void**)&h_C_result, cov_size)); 

    cargarImagenesHost(h_dataset, num_images, n, width, height, channels);

    checkCudaError(cudaEventRecord(start));

    int images_per_batch = num_images / S;
    size_t batch_size_bytes = (size_t)images_per_batch * n * sizeof(float);

    float *d_mu, *d_C_global;
    checkCudaError(cudaMalloc(&d_mu, n * sizeof(float)));
    checkCudaError(cudaMemset(d_mu, 0, n * sizeof(float)));
    checkCudaError(cudaMalloc(&d_C_global, cov_size));
    checkCudaError(cudaMemset(d_C_global, 0, cov_size)); 

    std::vector<float*> d_dataset_batches(S), d_centered_batches(S);
    std::vector<cudaStream_t> streams(S);
    for (int i = 0; i < S; i++) {
        checkCudaError(cudaMalloc(&d_dataset_batches[i], batch_size_bytes));
        checkCudaError(cudaMalloc(&d_centered_batches[i], batch_size_bytes));
        checkCudaError(cudaStreamCreate(&streams[i])); 
    }

    int threads1D = 256;

    // --- FASE 1: Obtener el Vector Promedio Global ---
    for (int s = 0; s < S; s++) {
        float* h_batch_ptr = h_dataset + (s * images_per_batch * n);
        checkCudaError(cudaMemcpyAsync(d_dataset_batches[s], h_batch_ptr, batch_size_bytes, cudaMemcpyHostToDevice, streams[s])); 
        
        kernelSumarLote<<<(n + threads1D - 1) / threads1D, threads1D, 0, streams[s]>>>(d_dataset_batches[s], d_mu, images_per_batch, n);
    }
    
    for (int s = 0; s < S; s++) {
        checkCudaError(cudaStreamSynchronize(streams[s]));
    }
    kernelFinalizarPromedio<<<(n + threads1D - 1) / threads1D, threads1D>>>(d_mu, num_images);
    checkCudaError(cudaDeviceSynchronize());

    // --- FASE 2: Centrado y Computo de Covarianza Tiled ---
    dim3 threads2D(TILE_DIM, TILE_DIM); 
    dim3 blocks2D((n + TILE_DIM - 1) / TILE_DIM, (n + TILE_DIM - 1) / TILE_DIM);

    for (int s = 0; s < S; s++) {
        int total_pixels_batch = images_per_batch * n;
        kernelCentrarDatos<<<(total_pixels_batch + threads1D - 1) / threads1D, threads1D, 0, streams[s]>>>(d_dataset_batches[s], d_mu, d_centered_batches[s], images_per_batch, n);
        kernelAcumularCovarianzaTiled<<<blocks2D, threads2D, 0, streams[s]>>>(d_centered_batches[s], d_C_global, images_per_batch, n);
    }

    for (int s = 0; s < S; s++) {
        checkCudaError(cudaStreamSynchronize(streams[s]));
    }

    int total_elements_C = n * n;
    kernelFinalizarCovarianza<<<(total_elements_C + threads1D - 1) / threads1D, threads1D>>>(d_C_global, num_images, n);
    checkCudaError(cudaDeviceSynchronize());

    // Transferir la gran matriz resultante de vuelta a la RAM del sistema
    checkCudaError(cudaMemcpy(h_C_result, d_C_global, cov_size, cudaMemcpyDeviceToHost));

    checkCudaError(cudaEventRecord(stop));
    checkCudaError(cudaEventSynchronize(stop));
    
    float ms = 0; 
    checkCudaError(cudaEventElapsedTime(&ms, start, stop));

    std::cout << "Streams: " << S << " | Tiempo Total (Copia + Computo): " << ms << " ms" << std::endl;

    for(int i = 0; i < S; i++) { 
        checkCudaError(cudaFree(d_dataset_batches[i])); 
        checkCudaError(cudaFree(d_centered_batches[i])); 
        checkCudaError(cudaStreamDestroy(streams[i])); 
    }
    checkCudaError(cudaFree(d_mu)); 
    checkCudaError(cudaFree(d_C_global)); 
    checkCudaError(cudaFreeHost(h_dataset)); 
    checkCudaError(cudaFreeHost(h_C_result));
    checkCudaError(cudaEventDestroy(start)); 
    checkCudaError(cudaEventDestroy(stop));
    
    return 0;
}