// Triangle counting algorithm implementation on GPU using CUDA C in CSR format

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

__global__ void count_triangles(uint32_t *IA_d,
                                uint32_t *JA_d,
                                int *delta_d,
                                int *IA_size_d,
                                int *JA_size_d){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int N = *IA_size_d - 1;
    // printf("\n size: \t %d", N);

    if (i > 0 && i < (N-1)){
        // printf("\n i: \t %d", i);
        uint32_t *a10T_row = IA_d + i;
        uint32_t *A20_row = IA_d + i + 1;

        uint32_t num_nnz_curr_row_a10T =  *A20_row - *a10T_row;

        uint32_t *a10T_start_col = JA_d + *a10T_row;
        uint32_t *a10T_end_col = a10T_start_col;

        // DEBUGGING TOOL
        // printf("\nthis is the %d th thread\n", i); 
        // printf("\nand a10T_row value is %d\n", *a10T_row);
        // printf("\n a10T_start_col \t %d", a10T_start_col);
        // printf("\n *a10T_start_col \t %d", *a10T_start_col);

        while (*a10T_end_col < i && (a10T_end_col - a10T_start_col) < num_nnz_curr_row_a10T -1){
            a10T_end_col++;
        }

        if (*a10T_end_col > i || (a10T_end_col - a10T_start_col) == num_nnz_curr_row_a10T){
            a10T_end_col--;
        }

        uint32_t *a12T_start_col = a10T_end_col + 1;
        uint32_t *a12T_end_col = a10T_start_col + num_nnz_curr_row_a10T - 1;

        uint32_t num_nnz_a12T = a12T_end_col - a12T_start_col + 1;
        uint32_t num_nnz_a10T = num_nnz_curr_row_a10T - num_nnz_a12T;

        // Adding number of triangles for each iteration 
        for(uint32_t k = 0;  k < num_nnz_a12T; k++){
            uint32_t a12T_select_location = *(a12T_start_col+k) - *a12T_start_col;
            uint32_t selected_row = *a12T_start_col + a12T_select_location - (i+1);
            uint32_t *A_row = A20_row + selected_row;
            uint32_t num_nnz_row_A20 = *(A_row + 1) - *A_row;

            uint32_t n = 0;
            uint32_t m = 0;
            while(n < num_nnz_a10T && m < num_nnz_row_A20){
                uint32_t *A20_u_col = JA_d + *A_row + m;
                uint32_t *a10T_u_col = a10T_start_col + n;
                if (*A20_u_col == *a10T_u_col){
                    atomicAdd(delta_d, 1);
                    m++;
                    n++;
                }
                else if (*A20_u_col > *a10T_u_col){
                    n++;
                }
                else {
                    m ++;
                }
            }
        }
    }
}
int main(){

    FILE *IA_ptr;
    FILE *JA_ptr;
    int IA_size[1];
    int JA_size[1];

    IA_ptr = fopen("/mnt/large/graph-datasets/ca-AstroPh_adj_IA.txt.bin", "rb");
    JA_ptr = fopen("/mnt/large/graph-datasets/ca-AstroPh_adj_JA.txt.bin", "rb");
    fread(IA_size, sizeof(IA_size), 1, IA_ptr);
    fread(JA_size, sizeof(JA_size), 1, JA_ptr);
    
    // STORE MATRICES IN HEAP MEMORY USING MALLOC
    uint32_t *IA = (uint32_t *)malloc(IA_size[0]*sizeof(uint32_t));
    uint32_t *JA = (uint32_t *)malloc(JA_size[0]*sizeof(uint32_t));
    size_t IA_size_t = IA_size[0];
    size_t JA_size_t = JA_size[0];
    fseek(IA_ptr, 4, SEEK_SET);
    fseek(JA_ptr, 4, SEEK_SET);
    fread(IA, sizeof(IA), IA_size_t, IA_ptr);
    fread(JA, sizeof(JA), JA_size_t, JA_ptr);

    for(int i = 0; i<10; i++){
        printf("\n %d", IA[i]);
    }

    //test case 1
    // uint32_t IA[] = {0,3,6,8,12,14};
    // uint32_t JA[] = {1,2,3,0,3,4,0,3,0,1,2,4,1,3};
    // int IA_size[1] = {6};
    // int JA_size[1] = {14};

    // test case 2
    // uint32_t IA[] = {0,4,6,10,12,17,19,22};
    // uint32_t JA[] = {1,2,4,6,0,2,0,1,3,4,2,4,0,2,3,5,6,4,6,0,4,5};
    // int IA_size[1] = {8};
    // int JA_size[1] = {22};

    //Allocatee Host Memory
    uint32_t delta[1] = {0};
    uint32_t *IA_d, *JA_d;
    int *delta_d, *IA_size_d, *JA_size_d;

    //Dynamically Allocate Device Memory
    cudaMalloc((void**) &IA_d, IA_size[0]*sizeof(uint32_t));
    cudaMalloc((void**) &JA_d, JA_size[0]*sizeof(uint32_t));
    cudaMalloc((void**) &delta_d, sizeof(int));
    cudaMalloc((void**) &IA_size_d, sizeof(int));
    cudaMalloc((void**) &JA_size_d, sizeof(int));

    //Copy host memory to device memory
    cudaMemcpy(IA_d, IA, IA_size[0]*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(JA_d, JA, JA_size[0]*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(delta_d, delta, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(IA_size_d, IA_size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(JA_size_d, JA_size, sizeof(int), cudaMemcpyHostToDevice);

    //execute kernel
    int threadsPerBlock = 32;
    int blocksPerGrid = IA_size[0] / 32 + 1;
    count_triangles<<<blocksPerGrid,threadsPerBlock>>>(IA_d, JA_d, delta_d, IA_size_d, JA_size_d);

    //Write GPU results in device memory back to host memory
    cudaMemcpy(delta, delta_d, sizeof(int), cudaMemcpyDeviceToHost);

    printf("\ndelta is %d\n", delta[0]);

    //Free device memory
    cudaFree(IA_d);
    cudaFree(JA_d);
    cudaFree(delta_d);
    cudaFree(IA_size_d);
    cudaFree(JA_size_d);

    //Free host memory
    free(IA);
    free(JA);


}