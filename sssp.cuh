//
// Created by gxl on 2021/1/6.
//

#ifndef PTGRAPH_SSSP_CUH
#define PTGRAPH_SSSP_CUH
#include "common.cuh"
void conventionParticipateSSSP(uint sourceNodeSample, string ssspPath);
void ssspShare(uint sourceNodeSample, string ssspPath);
void ssspOpt(uint sourceNodeSample, string ssspPath, float adviseK);
void ssspOptSwap();
long ssspCaculateInShare(uint testNumNodes, uint testNumEdge, uint *nodePointersI, EdgeWithWeight *edgeList,
                         uint sourceNode);
long
ssspCaculateCommonMemoryInnerAsync(uint testNumNodes, uint testNumEdge, uint *nodePointersI, EdgeWithWeight *edgeList,
                                   uint sourceNode, float adviseK);

long
ssspCaculateCommonMemoryInnerAsyncVisitRecord(uint testNumNodes, uint testNumEdge, uint *nodePointersI, EdgeWithWeight *edgeList,
                                              uint sourceNode, float adviseK);
long
ssspCaculateUVM(uint testNumNodes, uint testNumEdge, uint *nodePointersI, EdgeWithWeight *edgeList,
                uint sourceNode);
void ssspShareTrace(uint sourceNodeSample, string ssspPath);
long ssspCaculateInShareTrace(uint testNumNodes, uint testNumEdge, uint *nodePointersI, EdgeWithWeight *edgeList,
                              uint sourceNode);
long
ssspCaculateCommonMemoryInnerAsyncRandom(uint testNumNodes, uint testNumEdge, uint *nodePointersI, EdgeWithWeight *edgeList,
                                         uint sourceNode, float adviseK);
#endif //PTGRAPH_SSSP_CUH
