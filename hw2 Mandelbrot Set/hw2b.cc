/* final: {68 125.16} --> {68 124.90} --> {68 124.89} --> mpi: {68 116.03} --> openmpi: {68 89.16} --> {68 83.55} */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>
#include <immintrin.h>
const char* filename;
int iters, height, width;
int* image;
void write_png()
{
    FILE *fp = fopen(filename, "wb");
    // assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    // assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    // assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_FILTER_NONE);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 0);
    size_t row_size = 3 * width * sizeof(png_byte);
    for (int y = 0; y < height; ++y)
    {
        png_bytep row = (png_bytep)calloc(1, row_size);
        for (int x = 0; x < width; ++x)
        {
            int p = image[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters)
            {
                color[0] = (p&16) ? 240 :(p & 15) << 4;
                color[1] = color[2] = (p&16) ? (p & 15) << 4 : color[1];
                
            }
        }
        png_write_row(png_ptr, row);
        free(row);
    }
    
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}


int main(int argc, char **argv)
{
    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* detect how many CPUs are available */
    // cpu_set_t cpu_set;
    // sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    // int num_cpus = CPU_COUNT(&cpu_set);

    /* argument parsing */
    // assert(argc == 9);
    filename = argv[1];
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);
    iters = strtol(argv[2], 0, 10);
    

    /* allocate memory for the local image slice */
    int heightModSize = height % size; 
    int heightMulSize = height / size;
    int SumModMul = heightMulSize + heightModSize;
    int local_height = (rank == size - 1) ? SumModMul : heightMulSize;
    int other_local_height = rank * heightMulSize;
    int num_threads = omp_get_num_threads();
    int thread_rank = omp_get_thread_num();
    int end = local_height;
    int *local_image = (int *)malloc(width * local_height * sizeof(int));
    // assert(local_image);
    int endNum_threads = end / num_threads;
    unsigned long long thread_start = thread_rank * endNum_threads;
    unsigned long long thread_end = (thread_rank == num_threads - 1) ? end : (thread_start + endNum_threads);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int j = thread_start; j < thread_end; ++j)
    // for (int j = thread_rank; j < local_height; j += num_threads)
    {
        double y0 = (j + other_local_height) * ((upper - lower) / height) + lower;
        __m512d x0_vec = _mm512_set1_pd(left), x_offset = _mm512_set1_pd((right - left) / width);
        int i = 0;
        for (; i < width - 7; i += 8)
        {
            // _mm512_setzero_pd
            __m512d x0 = _mm512_set_pd(
                i+7,
                i+6,
                i+5,
                i+4,
                i+3,
                i+2,
                i+1,
                i
            );
            x0 = _mm512_add_pd(x0_vec, _mm512_mul_pd(x0, x_offset));
            __m512d x = _mm512_setzero_pd(), y = _mm512_setzero_pd(), length_squared = _mm512_setzero_pd(), two = _mm512_set1_pd(2.0), four = _mm512_set1_pd(4.0);
            __m256i one = _mm256_set1_epi32(1.0), iter_vec = _mm256_set1_epi32(iters), repeats = _mm256_setzero_si256(); // int: 32 bits * 8 = 256
            __mmask8 mask = 0xFF;
            __m512d xx = _mm512_mul_pd(x, x), yy = _mm512_mul_pd(y, y);
            // _mm256_cmplt_pd_mask(repeats, iters) // repeats < iters: set to 1
            while (mask)
            {
                // __m512d temp = _mm512_mul_pd(xySum, xySum); // x * x - y * y
                __m512d temp = _mm512_add_pd(_mm512_sub_pd(xx, yy), x0);  // x * x - y * y
                y = _mm512_add_pd(_mm512_mul_pd(two, _mm512_mul_pd(x, y)), _mm512_set1_pd(y0)); // 2 * x * y + y0
                x = temp;
                xx = _mm512_mul_pd(x, x), yy = _mm512_mul_pd(y, y);
                length_squared = _mm512_add_pd(xx, yy); // x * x + y * y
                repeats = _mm256_mask_add_epi32(repeats, mask, repeats, one); // repeats ++;
                // mask = _mm256_cmplt_epi32_mask(_mm256_mask_add_epi32(repeats, mask, repeats, one), iter_vec) & _mm512_cmplt_pd_mask(length_squared, four);
                mask = _mm256_cmplt_epi32_mask(repeats, iter_vec) & _mm512_cmplt_pd_mask(length_squared, four);
            }
            
            // _mm256_storeu_epi32(&local_image[j * width + i], repeats);
            // Store the iteration counts for the 8 elements
            int temp[8];
            _mm256_store_epi32(temp, repeats);
            for (int k = 0; k < 8; ++k)
            {
                local_image[j * width + i + k] = temp[k];
            }
        }
        for (; i < width; i++)
        {
            double x0 = left + i * ((right - left) / width), x = 0.0, y = 0.0, length_squared = 0.0;
            int repeats = 0;
            // bool mask = 1;
            // double xx=0, yy=0;
            while (length_squared < 4.0 && repeats < iters)
            {
                double temp = x*x - y*y + x0;
                // xy = x*y;
                y = 2.0 * x*y + y0;
                x = temp;
                length_squared = x*x + y*y;
                repeats++;
            }
            local_image[j * width + i] = repeats;
        }
    }
    /* gather all slices to rank 0 using MPI_Gatherv */
    int *all_h = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    image = NULL;
    for (int i = 0; i < size; ++i)
    {
        int new_local_height = (i == size - 1) ? SumModMul : heightMulSize;
        all_h[i] = new_local_height * width;
        displs[i] = i * heightMulSize * width;
    }
    if (rank == 0)
    {
        image = (int *)malloc(width * height * sizeof(int));
        // assert(image);
    }
    MPI_Gatherv(local_image, local_height * width, MPI_INT, image, all_h, displs, MPI_INT, 0, MPI_COMM_WORLD);

    // /* draw and cleanup */
    if (rank == 0)
    {
        write_png();
        // free(image);
    }

   
    /* Finalize MPI */
    MPI_Finalize();
    

    return 0;
}
