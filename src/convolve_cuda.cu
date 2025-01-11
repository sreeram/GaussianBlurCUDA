#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <stdexcept>
#include <cuda_runtime.h>
#include <stdio.h>

using namespace std;

// Note the correct CUDA kernel qualifier syntax
__global__ void convolve2D_gpu(const float *img, const float *kernel, float *output, 
                              int img_w, int img_h, int kernel_size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < img_w && row < img_h) {
        float sum = 0.0f;
        int kernel_radius = kernel_size / 2;
        
        // Corrected kernel application
        for (int i = -kernel_radius; i <= kernel_radius; i++) {
            for (int j = -kernel_radius; j <= kernel_radius; j++) {
                int img_row = row + i;
                int img_col = col + j;
                
                if (img_row >= 0 && img_row < img_h && img_col >= 0 && img_col < img_w) {
                    int kernel_idx = (i + kernel_radius) * kernel_size + (j + kernel_radius);
                    sum += img[img_row * img_w + img_col] * kernel[kernel_idx];
                }
            }
        }
        output[row * img_w + col] = sum;
    }
}

int main() {
    ifstream file("gaussian_blur_input.txt");
    if (!file.is_open()) {
        cerr << "Error: Could not open input file." << endl;
        return 1;
    }

    int ker_rows, ker_cols;
    file >> ker_rows >> ker_cols;

    // Read kernel matrix from file as 1d array
    vector<float> kernel(ker_rows * ker_cols);
    for (int i = 0; i < ker_rows * ker_cols; i++) {
        file >> kernel[i];
    }

    int img_rows, img_cols;
    file >> img_rows >> img_cols;

    // Changed to float vector to match kernel input type
    vector<float> img(img_rows * img_cols);
    for (int i = 0; i < img_rows * img_cols; i++) {
        float pixel_value;
        file >> pixel_value;
        img[i] = pixel_value;
    }

    file.close();

    vector<float> output(img_rows * img_cols);

    // Allocate memory on device
    float *d_img, *d_kernel, *d_output;
    cudaMalloc(&d_img, img_rows * img_cols * sizeof(float));
    cudaMalloc(&d_kernel, ker_rows * ker_cols * sizeof(float));
    cudaMalloc(&d_output, img_rows * img_cols * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_img, img.data(), img_rows * img_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.data(), ker_rows * ker_cols * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize output array to zeros
    cudaMemset(d_output, 0, img_rows * img_cols * sizeof(float));

    // Define block size and grid size
    dim3 block_size(16, 16);  // Reduced block size for better occupancy
    dim3 grid_size((img_cols + block_size.x - 1) / block_size.x, 
                   (img_rows + block_size.y - 1) / block_size.y);

    // Start timer
    auto start = chrono::high_resolution_clock::now();

    // Call kernel
    convolve2D_gpu<<<grid_size, block_size>>>(d_img, d_kernel, d_output, 
                                             img_cols, img_rows, ker_rows);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "Kernel launch error: " << cudaGetErrorString(err) << endl;
        return 1;
    }

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Stop timer
    auto end = chrono::high_resolution_clock::now();

    // Copy data from device to host
    cudaMemcpy(output.data(), d_output, img_rows * img_cols * sizeof(float), 
               cudaMemcpyDeviceToHost);

    // Write output and time to file
    ofstream out_file("gaussian_blur_cuda_output.txt");
    if (!out_file.is_open()) {
        cerr << "Error: Could not open output file." << endl;
        return 1;
    }

    out_file << img_rows << " " << img_cols << endl;
    for (int i = 0; i < img_rows * img_cols; i++) {
        out_file << output[i] << " ";
    }
    out_file << endl;

    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    out_file << duration.count() << " microseconds" << endl;

    out_file.close();

    // Free memory
    cudaFree(d_img);
    cudaFree(d_kernel);
    cudaFree(d_output);

    return 0;
}

