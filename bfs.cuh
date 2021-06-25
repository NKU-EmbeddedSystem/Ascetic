//
// Created by gxl on 2020/12/30.
//

#ifndef PTGRAPH_BFS_CUH
#define PTGRAPH_BFS_CUH

#include "common.cuh"

void conventionParticipateBFS(string bfsPath, int sampleSourceNode);
void bfsShare(string bfsPath, int sampleSourceNode);
void bfsOpt(string bfsPath, int sampleSourceNode, float adviseK);
long bfsCaculateInShare(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList, uint sourceNode);

long
bfsCaculateInShareReturnValue(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList, uint sourceNode,
                              uint **bfsValue, int index);

long
bfsCaculateInAsyncNoUVMSwap(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList, uint sourceNode);

long
bfsCaculateInAsyncNoUVM(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList, uint sourceNode, float adviseK);

long
bfsCaculateInAsyncNoUVMVisitRecord(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList, uint sourceNode,
                                   float adviseK);
long bfsCaculateInShareTrace(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList, uint sourceNode);
void bfsShareTrace(string bfsPath, int sampleSourceNode);
long
bfsCaculateInAsyncNoUVMRandom(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList, uint sourceNode,
                              float adviseK);
#endif //PTGRAPH_BFS_CUH
