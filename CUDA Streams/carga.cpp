#define cimg_display 0
#include "CImg.h"
#include <iostream>
#include <cstdio>
#include <vector>

using namespace cimg_library;

void obtenerDimensionesHost(int& w, int& h, int& c) {
    try {
        CImg<unsigned char> img("dataset/0801x4.bmp");
        w = img.width(); 
        h = img.height(); 
        c = img.spectrum();
    } catch (CImgException &e) {
        std::cerr << "CRITICO: No se pudo abrir 'dataset/0801x4.bmp'." << std::endl;
        exit(EXIT_FAILURE);
    }
}

void cargarImagenesHost(float* h_dataset, int num_images, int n, int width, int height, int channels) {
    std::cout << "Cargando " << num_images << " imagenes (0801-0900) en la memoria RAM..." << std::endl;
    
    // Calculamos el tamaño real de la imagen en disco
    int n_real = width * height * channels;

    for (int k = 0; k < num_images; k++) {
        char buf[64]; 
        std::snprintf(buf, 64, "dataset/%04dx4.bmp", k + 801);
        
        try {
            CImg<unsigned char> img(buf);
            
            // Copiamos solo hasta 'n' elementos (el límite seguro asignado por CUDA)
            for (int i = 0; i < n; i++) {
                if (i < n_real) {
                    h_dataset[k * n + i] = (float)img.data()[i];
                } else {
                    h_dataset[k * n + i] = 0.0f; // Padding por si acaso
                }
            }

        } catch (CImgException &e) {
            std::cerr << "Error al cargar el archivo: " << buf << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    std::cout << "Carga en Host completada con exito!" << std::endl;
}