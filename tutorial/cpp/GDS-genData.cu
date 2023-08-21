// To compile this sample code:
//
// nvcc gds_helloworld.cxx -o gds_helloworld -lcufile
//
// Set the environment variable TESTFILE
// to specify the name of the file on a GDS enabled filesystem
//
// Ex:   TESTFILE=/mnt/gds/gds_test ./gds_helloworld
//
//
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#include <iostream>

#include <cuda_runtime.h>
#include <nvcufile.h>

#include "helper_cuda.h"

using namespace std;

int main(void) {
    cudaError_t cuda_result;
    ssize_t ret;
    CUfileError_t status;

    off_t file_offset = 0x0;
    off_t devPtr_offset = 0x0;
    ssize_t IO_size = 4096ULL;
    size_t buff_size = IO_size + 0x0;

    int d = 64;      // dimension
    int nb = 100000; // database size
    int nq = 10000;  // nb of queries
    float* host_data = new float[d * nb];

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    printf("Filling memory.\n");
    float* xb = new float[d * nb];
    float* xq = new float[d * nq];

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.;
    }

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = distrib(rng);
        xq[d * i] += i / 1000.;
    }

    printf("Allocating CUDA buffer\n");
    float *dev_xb, *dev_xq;
    checkCudaErrors(cudaMalloc(&dev_xb, d * nb * sizeof(float)));
    checkCudaErrors(cudaMalloc(&dev_xq, d * nq * sizeof(float)));

    checkCudaErrors(cudaMemcpy(dev_xb, xb, d * nb * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_xq, xq, d * nq * sizeof(float), cudaMemcpyHostToDevice));

    int base_fd = open("/mnt/nvmetest/faiss_data_trained.c", O_CREAT | O_RDWR | O_DIRECT, 0644);
    if(base_fd < 0) {
        printf("file open errno %d\n", errno);
        return -1;
    }
    int query_fd = open("/mnt/nvmetest/faiss_data_query.c", O_CREAT | O_RDWR | O_DIRECT, 0644);
    if(query_fd < 0) {
        printf("file open errno %d\n", errno);
        return -1;
    }

    printf("Opening cuFileDriver.\n");
    status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
        printf(" cuFile driver failed to open \n");
        goto cufile_open_failed;
    }

    printf("Registering cuFile handle.\n");
    CUfileDescr_t cf_descr;
    CUfileHandle_t cf_handle;
    memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = base_fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    status = cuFileHandleRegister(&cf_handle, &cf_descr);
    if (status.err != CU_FILE_SUCCESS) {
        std::cerr << "cuFileHandleRegister base_fd " << base_fd << " status " << status.err << std::endl;
        goto handle_register_failed;
    }

    printf(" Registering Buffer of %lu bytes and %lu bytes.\n",
        d * nb * sizeof(float), d * nq * sizeof(float));
    status = cuFileBufRegister(dev_xb, d * nb * sizeof(float), 0);
    if (status.err != CU_FILE_SUCCESS) {
        printf("buffer registration failed %d\n", status.err);
        goto register_failed;
    }
    status = cuFileBufRegister(dev_xq, d * nq * sizeof(float), 0);
    if (status.err != CU_FILE_SUCCESS) {
        printf("buffer registration failed %d\n", status.err);
        goto register_failed;
        return -1;
    }

    // perform write operation directly from GPU mem to file
    printf("Writing buffer to file.\n");
    ret = cuFileWrite(cf_handle, dev_xb, d * nb * sizeof(float), 0, 0);
    if (ret < 0 || ret != d * nb * sizeof(float)) {
        printf("cuFileWrite failed %zu\n", ret);
        goto write_fail;
    }
    
    // print file data
    checkCudaErrors(cudaMemcpy(host_data, dev_xb, d * nb * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < d * nb; i++) {
        if (i / d >= 100) {
            printf("...\n");
            break;
        }
        printf("%.2f ", host_data[i]);
        if ((i + 1) % d == 0) {
            printf("\n");
        }
    }

    // release the GPU memory pinning
    printf("Releasing cuFile buffer.\n");
    status = cuFileBufDeregister(dev_xb);
    if (status.err != CU_FILE_SUCCESS) {
        std::cerr << "buffer deregister failed" << std::endl;
        cudaFree(dev_xb);
        cuFileHandleDeregister(cf_handle);
        close(base_fd);
        return -1;
    }
    status = cuFileBufDeregister(dev_xq);
    if (status.err != CU_FILE_SUCCESS) {
        std::cerr << "buffer deregister failed" << std::endl;
        cudaFree(dev_xb);
        cuFileHandleDeregister(cf_handle);
        close(base_fd);
        return -1;
    }

    printf("Freeing CUDA buffer.\n");
    checkCudaErrors(cudaFree(dev_xb));
    checkCudaErrors(cudaFree(dev_xq));

        // deregister the handle from cuFile
        cout << "Releasing file handle. " << std::endl;
        (void) cuFileHandleDeregister(cf_handle);
        close(base_fd);

        // release all cuFile resources
        cout << "Closing File Driver." << std::endl;
        (void) cuFileDriverClose();
        cout << std::endl;

    return 0;

write_fail:
register_failed:
    cuFileHandleDeregister(cf_handle);
    cudaFree(dev_xb);
    cudaFree(dev_xq);
handle_register_failed:
    (void) cuFileDriverClose();
cufile_open_failed:
    close(base_fd);
    close(query_fd);

    return -1;
}
