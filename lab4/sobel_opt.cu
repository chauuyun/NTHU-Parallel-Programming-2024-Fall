#include <iostream>
#include <cstdlib>
#include <cassert>
#include <zlib.h>
#include <png.h>
#include <cuda_runtime.h>
#include <math.h>

#define Z 2
#define Y 5
#define X 5
#define xBound X / 2
#define yBound Y / 2
#define SCALE 8
int read_png(const char* filename, unsigned char** image, unsigned* height, unsigned* width, unsigned* channels) {

    unsigned char sig[8];
    FILE* infile;
    infile = fopen(filename, "rb");

    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8))
        return 1;   /* bad signature */

    png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
        return 4;   /* out of memory */
  
    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4;   /* out of memory */
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32  i, rowbytes;
    png_bytep  row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int) png_get_channels(png_ptr, info_ptr);

    if ((*image = (unsigned char *) malloc(rowbytes * *height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }

    for (i = 0;  i < *height;  ++i)
        row_pointers[i] = *image + i * rowbytes;
    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width, 
               const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_FILTER_NONE);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 0);

    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++ i) {
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}


__constant__ char mask[Z][Y][X] = {
    {
        { -1, -4, -6, -4, -1 },
        { -2, -8, -12, -8, -2 },
        { 0, 0, 0, 0, 0 },
        { 2, 8, 12, 8, 2 },
        { 1, 4, 6, 4, 1 }
    },
    {
        { -1, -2, 0, 2, 1 },
        { -4, -8, 0, 8, 4 },
        { -6, -12, 0, 12, 6 },
        { -4, -8, 0, 8, 4 },
        { -1, -2, 0, 2, 1 }
    }
};
__global__ void sobel(unsigned char *s, unsigned char *t, unsigned height, unsigned width, unsigned channels) {
    // Block and thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * (blockDim.x - 4) + threadIdx.x - 2;
    int y = blockIdx.y * (blockDim.y - 4) + threadIdx.y - 2;

    // Shared memory size (blockDim + kernel overlap)
    __shared__ unsigned char sR[20][20];
    __shared__ unsigned char sG[20][20];
    __shared__ unsigned char sB[20][20];

    // Initialize shared memory
    unsigned char R = 0, G = 0, B = 0;

    if (x >= 0 && x < width && y >= 0 && y < height) {
        int idx = channels * (width * y + x);
        B = s[idx + 0];
        G = s[idx + 1];
        R = s[idx + 2];
    }

    // Load data into shared memory
    int shared_x = tx;
    int shared_y = ty;
    sR[shared_y][shared_x] = R;
    sG[shared_y][shared_x] = G;
    sB[shared_y][shared_x] = B;

    __syncthreads();

    // Only process the central part of the block
    if (tx >= 2 && tx < (blockDim.x - 2) && ty >= 2 && ty < (blockDim.y - 2)) {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            float val[Z][3] = {0};

            // Apply convolution
            
            for (int v = -yBound; v <= yBound; ++v) {
                for (int u = -xBound; u <= xBound; ++u) {
                    int smem_x = shared_x + u;
                    int smem_y = shared_y + v;

                    unsigned char r = sR[smem_y][smem_x];
                    unsigned char g = sG[smem_y][smem_x];
                    unsigned char b = sB[smem_y][smem_x];

                    val[0][0] += b * mask[0][v + yBound][u + xBound];
                    val[0][1] += g * mask[0][v + yBound][u + xBound];
                    val[0][2] += r * mask[0][v + yBound][u + xBound];
                    val[1][0] += b * mask[1][v + yBound][u + xBound];
                    val[1][1] += g * mask[1][v + yBound][u + xBound];
                    val[1][2] += r * mask[1][v + yBound][u + xBound];
                }
            }
            

            // Compute magnitude
            float totalB = 0.;
            float totalG = 0.;
            float totalR = 0.;

            totalB += val[0][0] * val[0][0];
            totalG += val[0][1] * val[0][1];
            totalR += val[0][2] * val[0][2];
            totalB += val[1][0] * val[1][0];
            totalG += val[1][1] * val[1][1];
            totalR += val[1][2] * val[1][2];
            

            totalB = sqrtf(totalB) / SCALE;
            totalG = sqrtf(totalG) / SCALE;
            totalR = sqrtf(totalR) / SCALE;

            const unsigned char cB = (totalB > 255.0) ? 255 : static_cast<unsigned char>(totalB);
            const unsigned char cG = (totalG > 255.0) ? 255 : static_cast<unsigned char>(totalG);
            const unsigned char cR = (totalR > 255.0) ? 255 : static_cast<unsigned char>(totalR);

            int output_idx = channels * (width * y + x);
            t[output_idx + 0] = cB;
            t[output_idx + 1] = cG;
            t[output_idx + 2] = cR;
        }
    }
}

int main(int argc, char **argv) {
    assert(argc == 3);
    unsigned height, width, channels;
    unsigned char *src = NULL, *dst;
    unsigned char *dsrc, *ddst;

    // Read the image
    if (read_png(argv[1], &src, &height, &width, &channels)) {
        std::cerr << "Error in read png" << std::endl;
        return -1;
    }

    dst = (unsigned char *)malloc(height * width * channels * sizeof(unsigned char));
    cudaHostRegister(src, height * width * channels * sizeof(unsigned char), cudaHostRegisterDefault);

    // Allocate device memory
    cudaMalloc(&dsrc, height * width * channels * sizeof(unsigned char));
    cudaMalloc(&ddst, height * width * channels * sizeof(unsigned char));

    // Copy source image to device
    cudaMemcpy(dsrc, src, height * width * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Set up execution configuration
    dim3 blockDim(20, 20);
    dim3 gridDim((width + (blockDim.x - 4) - 1) / (blockDim.x - 4), (height + (blockDim.y - 4) - 1) / (blockDim.y - 4));

    // Launch kernel
    sobel<<<gridDim, blockDim>>>(dsrc, ddst, height, width, channels);

    // Copy result back to host
    cudaMemcpy(dst, ddst, height * width * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Write the output image
    write_png(argv[2], dst, height, width, channels);

    // Free memory
    free(src);
    free(dst);
    cudaFree(dsrc);
    cudaFree(ddst);

    return 0;
}