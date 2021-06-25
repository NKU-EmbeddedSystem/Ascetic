//
// Created by gxl on 2021/3/24.
//

#ifndef PTGRAPH_CALCULATEOPT_CUH
#define PTGRAPH_CALCULATEOPT_CUH
#include "GraphMeta.cuh"

#include "gpu_kernels.cuh"
#include "TimeRecord.cuh"
void bfs_opt(string path, uint sourceNode, double adviseRate);
void cc_opt(string path, double adviseRate);
void sssp_opt(string path, uint sourceNode, double adviseRate);
void pr_opt(string path, double adviseRate);
#endif //PTGRAPH_CALCULATEOPT_CUH
