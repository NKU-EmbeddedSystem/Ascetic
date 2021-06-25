//
// Created by gxl on 2021/1/5.
//

#ifndef PTGRAPH_CC_CUH
#define PTGRAPH_CC_CUH
#include "common.cuh"
void conventionParticipateCC(string ccPath);
void ccShare(string ccPath);
void ccOpt(string ccPath, float adviseK);
void ccOptSwap();
long ccCaculateInShare(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList);
long ccCaculateCommonMemoryInnerAsync(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList, float adviseK);
void conventionParticipateCCInLong();
long ccCaculateCommonMemoryInnerAsyncRecordVisit(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList,
                                      float adviseK);
long ccCaculateInShareTrace(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList);
void ccShareTrace(string ccPath);
long ccCaculateCommonMemoryInnerAsyncRandom(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList,
                                            float adviseK);
#endif //PTGRAPH_CC_CUH
