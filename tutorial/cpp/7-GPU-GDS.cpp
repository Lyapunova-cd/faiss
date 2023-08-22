/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <random>

#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>

int main() {
    int d = 64;      // dimension
    int nb = 100000; // database size
    int nq = 10000;  // nb of queries

    faiss::gpu::StandardGpuResources res;

    // Using a flat index

#if defined USE_NVIDIA_GDS
    printf("use USE_NVIDIA_GDS flag\n");
#endif

    faiss::gpu::GpuIndexFlatL2 index_flat(&res, d);

    int base_fd = open("/mnt/nvmetest/faiss_data_trained.c", O_CREAT | O_RDWR | O_DIRECT, 0644);
    assert(base_fd > 0);
    int query_fd = open("/mnt/nvmetest/faiss_data_query.c", O_CREAT | O_RDWR | O_DIRECT, 0644);
    assert(query_fd > 0);
    printf("open training data file, fd %d\n", base_fd);
    printf("open query data file, fd %d\n", query_fd);

    printf("is_trained = %s\n", index_flat.is_trained ? "true" : "false");
    index_flat.add(nb, base_fd);
    printf("ntotal = %ld\n", index_flat.ntotal);

    int k = 4;

    // { // search xq
    //     long* I = new long[k * nq];
    //     float* D = new float[k * nq];

    //     index_flat.search(nq, xq, k, D, I);

    //     // print results
    //     printf("I (5 first results)=\n");
    //     for (int i = 0; i < 5; i++) {
    //         for (int j = 0; j < k; j++)
    //             printf("%5ld ", I[i * k + j]);
    //         printf("\n");
    //     }

    //     printf("I (5 last results)=\n");
    //     for (int i = nq - 5; i < nq; i++) {
    //         for (int j = 0; j < k; j++)
    //             printf("%5ld ", I[i * k + j]);
    //         printf("\n");
    //     }

    //     delete[] I;
    //     delete[] D;
    // }

    // // Using an IVF index

    // int nlist = 100;
    // faiss::gpu::GpuIndexIVFFlat index_ivf(&res, d, nlist, faiss::METRIC_L2);

    // assert(!index_ivf.is_trained);
    // index_ivf.train(nb, xb);
    // assert(index_ivf.is_trained);
    // index_ivf.add(nb, xb); // add vectors to the index

    // printf("is_trained = %s\n", index_ivf.is_trained ? "true" : "false");
    // printf("ntotal = %ld\n", index_ivf.ntotal);

    // { // search xq
    //     long* I = new long[k * nq];
    //     float* D = new float[k * nq];

    //     index_ivf.search(nq, xq, k, D, I);

    //     // print results
    //     printf("I (5 first results)=\n");
    //     for (int i = 0; i < 5; i++) {
    //         for (int j = 0; j < k; j++)
    //             printf("%5ld ", I[i * k + j]);
    //         printf("\n");
    //     }

    //     printf("I (5 last results)=\n");
    //     for (int i = nq - 5; i < nq; i++) {
    //         for (int j = 0; j < k; j++)
    //             printf("%5ld ", I[i * k + j]);
    //         printf("\n");
    //     }

    //     delete[] I;
    //     delete[] D;
    // }

    return 0;
}
