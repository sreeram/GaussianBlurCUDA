#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <stdexcept>
using namespace std;

__global__ void convolve2D_gpu(float *img, float *imgf, float *kernel, int Nx, int Ny, int kernel_size) {
    //local ID of each thread (withing block) 
    int tid = threadIdx.x;    
    
    //each block is assigned to a row of an image, iy index of y value                  
    int iy = blockIdx.x + (kernel_size - 1)/2;  
    
    //each thread is assigned to a pixel of a row, ix index of x value
    int ix = threadIdx.x + (kernel_size - 1)/2; 
    
    //idx global index (all blocks) of the image pixel 
    int idx = iy*Nx +ix;                        
    
    
    //total number of kernel elements
    int K2 = kernel_size*kernel_size; 
    
    //center of kernel in both dimensions          
    int center = (kernel_size -1)/2;		
    
    //Auxiliary variables
    int ii, jj;
    float sum = 0.0;

    
    /*
    Define a vector (float) sdata[] that will be hosted in shared memory,
    *extern* dynamic allocation of shared memory: kernel<<<blocks,threads,memory size to be allocated in shared memory>>>
    */  
    extern __shared__ float sdata[];         


    /*Transfer data frm GPU memory to shared memory 
    tid: local index, each block has access to its local shared memory
    e.g. 100 blocks -> 100 allocations/memory spaces
    Eeah block has access to the kernel coefficients which are store in shared memory
    Important: tid index must not exceed the size of the kernel*/

    if (tid<K2)
        sdata[tid] = kernel[tid];             
    
    
    
    
    //Important. Syncronize threads before performing the convolution.
    //Ensure that shared memory is filled by the tid threads
    
    __syncthreads();			  
                            
    
    
    
    /*
    Convlution of image with the kernel
    Each thread computes the resulting pixel value 
    from the convolution of the original image with the kernel;
    number of computations per thread = size_kernel^2
    The result is stored to imgf
    */
    
    if (idx<Nx*Ny){
        for (int ki = 0; ki<kernel_size; ki++)
        for (int kj = 0; kj<kernel_size; kj++){
        ii = kj + ix - center;
        jj = ki + iy - center;
        sum+=img[jj*Nx+ii]*sdata[ki*kernel_size + kj];
        }
    
        imgf[idx] = sum;
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

    // read kernel matrix from file as 2D vector of floats
    vector<vector<float>> kernel(ker_rows, vector<float>(ker_cols));
    for (int i = 0; i < ker_rows; i++) {
        for (int j = 0; j < ker_cols; j++) {
            file >> kernel[i][j];
        }
    }

    // flip the kernel matrix
    for (int i = 0; i < ker_rows / 2; i++) {
        for (int j = 0; j < ker_cols; j++) {
            swap(kernel[i][j], kernel[ker_rows - i - 1][ker_cols - j - 1]);
        }
    }

    // Convert kernel to 1D array
    vector<float> kernel_1d;
    for (int i = 0; i < ker_rows; i++) {
        for (int j = 0; j < ker_cols; j++) {
            kernel_1d.push_back(kernel[i][j]);
        }
    }    

    int img_rows, img_cols;
    file >> img_rows >> img_cols;

    // read image matrix from file as 1D vector of ints
    vector<float> image_1d(img_rows * img_cols);
    for (int i = 0; i < img_rows * img_cols; i++) {
        file >> image_1d[i];
    }

    file.close();

    // Allocate memory on the GPU
    float *d_image, *d_imagef, *d_kernel;
    cudaMalloc(&d_image, img_rows * img_cols * sizeof(float));
    cudaMalloc(&d_imagef, img_rows * img_cols * sizeof(float));
    cudaMalloc(&d_kernel, ker_rows * ker_cols * sizeof(float));

    // Copy data to the GPU
    cudaMemcpy(d_image, image_1d.data(), img_rows * img_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel_1d.data(), ker_rows * ker_cols * sizeof(float), cudaMemcpyHostToDevice);

    int Nblocks = img_rows - ker_rows + 1;
    int Nthreads = img_cols - ker_cols + 1;
    // Perform convolution
    auto start = chrono::high_resolution_clock::now();
    convolve2D_gpu<<<Nblocks, Nthreads, ker_rows * ker_cols * sizeof(float)>>>(d_image, d_imagef, d_kernel, img_cols, img_rows, ker_rows);
    cudaDeviceSynchronize();
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    // Copy result back to host
    vector<float> output(img_rows * img_cols);
    cudaMemcpy(output.data(), d_imagef, img_rows * img_cols * sizeof(float), cudaMemcpyDeviceToHost);

    // reshape output to 2D
    vector<vector<int>> output_2d(img_rows, vector<int>(img_cols));
    for (int i = 0; i < img_rows; i++) {
        for (int j = 0; j < img_cols; j++) {
            output_2d[i][j] = output[i * img_cols + j];
        }
    }

    // Write output and time to file
    ofstream out_file("gaussian_blur_cuda_output.txt");
    if (!out_file.is_open()) {
        cerr << "Error: Could not open output file." << endl;
        return 1;
    }

    out_file << output_2d.size() << " " << output_2d[0].size() << endl;
    for (int i = 0; i < output_2d.size(); i++) {
        for (int j = 0; j < output_2d[0].size(); j++) {
            out_file << output_2d[i][j] << " ";
        }
        out_file << endl;
    }
    out_file << duration.count() << endl;
    out_file.close();
}