#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <stdexcept>
using namespace std;

// Function to perform convolution
vector<vector<int>> convolve2D(const vector<vector<int>>& image, const vector<vector<float>>& kernel) {
    int matrixRows = image.size();
    int matrixCols = image[0].size();
    int kernelRows = kernel.size();
    int kernelCols = kernel[0].size();

    // Validate dimensions
    if (matrixRows < kernelRows || matrixCols < kernelCols) {
        throw invalid_argument("Kernel dimensions are larger than the matrix dimensions.");
    }

    // The output size after convolution
    int outputRows = matrixRows - kernelRows + 1;
    int outputCols = matrixCols - kernelCols + 1;

    // Create the output matrix
    vector<vector<int>> output(outputRows, vector<int>(outputCols, 0));

    // Perform convolution
    for (int i = 0; i < outputRows; i++) {
        for (int j = 0; j < outputCols; j++) {
            int sum = 0;
            // Loop over the kernel
            for (int m = 0; m < kernelRows; m++) {
                for (int n = 0; n < kernelCols; n++) {
                    sum += image[i + m][j + n] * kernel[m][n];
                }
            }
            output[i][j] = sum;
        }
    }

    return output;
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

    int img_rows, img_cols;
    file >> img_rows >> img_cols;

    // read image matrix from file as 2D vector of integers
    vector<vector<int>> image(img_rows, vector<int>(img_cols));
    for (int i = 0; i < img_rows; i++) {
        for (int j = 0; j < img_cols; j++) {
            file >> image[i][j];
        }
    }

    file.close();

    // Perform convolution
    auto start = chrono::high_resolution_clock::now();
    vector<vector<int>> output = convolve2D(image, kernel);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    // Write output and time to file
    ofstream out_file("gaussian_blur_output.txt");
    if (!out_file.is_open()) {
        cerr << "Error: Could not open output file." << endl;
        return 1;
    }

    out_file << output.size() << " " << output[0].size() << endl;
    for (int i = 0; i < output.size(); i++) {
        for (int j = 0; j < output[0].size(); j++) {
            out_file << output[i][j] << " ";
        }
        out_file << endl;
    }
    out_file << duration.count() << endl;
    out_file.close();
}

