#ifndef GPU_KERNELS_CUH
#define GPU_KERNELS_CUH

#include "range.hpp"
#include "globals.cuh"

using namespace util::lang;

// type alias to simplify typing...
template<typename T>
using step_range = typename range_proxy<T>::step_range_proxy;

template<typename T>
__device__
step_range<T> grid_stride_range(T begin, T end) {
    begin += blockDim.x * blockIdx.x + threadIdx.x;
    return range(begin, end).step(gridDim.x * blockDim.x);
}

template<typename Predicate>
__device__ void streamVertices(int vertices, Predicate p) {
    for (auto i : grid_stride_range(0, vertices)) {
        p(i);
    }
}

uint reduceBool(uint* resultD, bool* isActiveD, uint vertexSize, dim3 grid, dim3 block);
__device__ void reduceStreamVertices(int vertices, bool *rawData, uint *result);
__global__ void reduceByBool(uint vertexSize, bool *rawData, uint *result);

template <int blockSize> __global__ void reduceResult(uint *result);
__global__ void
bfs_kernel(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD, uint *valueD,
           bool *labelD);

__global__ void
cc_kernel(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD, uint *valueD,
          bool *labelD);

template <typename T>
__global__ void
sssp_kernel(uint activeNum, const uint *activeNodesD, const uint *nodePointersD, const uint *degreeD, EdgeWithWeight *edgeListD,
            uint *valueD,
            T *labelD);

template <typename T, typename E>
__global__ void
sssp_kernelDynamic(uint overloadStartNode, uint overloadNodeNum, const uint *overloadNodeListD,
                   const uint *degreeD,
                   uint *valueD,
                   T *isActiveD, const EdgeWithWeight *edgeListOverloadD,
                   const E *activeOverloadNodePointersD);

__global__ void
bfs_kernelOpt(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD, uint *valueD,
              bool *labelD, uint overloadNode, uint *overloadEdgeListD,
              uint *nodePointersOverloadD);

__global__ void
bfs_kernelStatic2Label(uint nodeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD,
                       uint *valueD,
                       uint *isActiveD1, uint *isActiveD2);

__global__ void
bfs_kernelDynamic2Label(uint overloadStartNode, uint overloadNodeNum, const uint *overloadNodeListD,
                        const uint *degreeD,
                        uint *valueD,
                        uint *isActiveD1, uint *isActiveD2, const uint *edgeListOverloadD,
                        const uint *activeOverloadNodePointersD);

template<typename T, typename E>
__global__ void
bfs_kernelDynamicPart(uint overloadStartNode, uint overloadNodeNum, const uint *overloadNodeListD,
                      const uint *degreeD,
                      uint *valueD,
                      T *isActiveD, const uint *edgeListOverloadD,
                      const E *activeOverloadNodePointersD);

__global__ void
setStaticAndOverloadNodePointer(uint vertexNum, uint *staticNodes, uint *overloadNodes, uint *overloadNodePointers,
                                uint *staticLabel, uint *overloadLabel,
                                uint *staticPrefix, uint *overloadPrefix, uint *degreeD);

__global__ void
sssp_kernelStaticSwapOpt2Label(uint activeNodesNum, const uint *activeNodeListD,
                               const uint *staticNodePointerD, const uint *degreeD,
                               EdgeWithWeight *edgeListD, uint *valueD, uint *isActiveD1, uint *isActiveD2,
                               bool *isFinish);

__global__ void
sssp_kernelDynamicSwap2Label(uint overloadStartNode, uint overloadNodeNum, const uint *overloadNodeListD,
                             const uint *degreeD,
                             uint *valueD,
                             uint *isActiveD1, uint *isActiveD2, const EdgeWithWeight *edgeListOverloadD,
                             const uint *activeOverloadNodePointersD, bool *finished);

template<class T>
__global__ void
bfs_kernelStatic(uint nodeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD, uint *valueD,
                 T *labelD);

__global__ void
bfs_kernelStaticSwap(uint nodeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD,
                     uint *valueD,
                     uint *labelD, bool *isInD);

__global__ void
bfs_kernelStaticSwap(uint nodeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD,
                     uint *valueD,
                     uint *labelD, bool *isInD, uint *fragmentRecordsD, uint fragment_size);

__global__ void
bfs_kernelStaticSwap(uint nodeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD,
                     uint *valueD,
                     uint *labelD, uint *fragmentRecordsD, uint fragment_size);

__global__ void
bfs_kernelStaticSwap(uint nodeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD,
                     uint *valueD,
                     uint *labelD, uint *fragmentRecordsD, uint fragment_size, uint maxpartionSize, uint testNumNodes);

__global__ void
bfs_kernelDynamic(uint activeNum, uint *activeNodesD, uint *degreeD, uint *valueD,
                  uint *labelD, uint overloadNode, uint *overloadEdgeListD,
                  uint *nodePointersOverloadD);

__global__ void
bfs_kernelDynamicSwap(uint activeNum, uint *activeNodesD, uint *degreeD, uint *valueD,
                      uint *labelD, uint *overloadEdgeListD,
                      uint *nodePointersOverloadD);

__global__ void
sssp_kernelOpt(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, EdgeWithWeight *edgeListD,
               uint *valueD,
               bool *labelD, uint overloadNode, EdgeWithWeight *overloadEdgeListD, uint *nodePointersOverloadD);
__global__ void
sssp_kernelDynamicUvm(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, EdgeWithWeight *edgeListD,
                      uint *valueD,
                      uint *labelD1, uint *labelD2);

__global__ void
cc_kernelOpt(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD, uint *valueD,
             bool *labelD, uint overloadNode, uint *overloadEdgeListD, uint *nodePointersOverloadD);

template <typename T, typename E>
__global__ void
cc_kernelDynamicSwap(uint overloadStartNode, uint overloadNodeNum, const uint *overloadNodeListD, const uint *degreeD,
                     uint *valueD,
                     T *isActiveD, const uint *edgeListOverloadD,
                     const E *activeOverloadNodePointersD);

__global__ void
cc_kernelDynamicSwap2Label(uint overloadStartNode, uint overloadNodeNum, const uint *overloadNodeListD,
                           const uint *degreeD,
                           uint *valueD,
                           uint *isActiveD1, uint *isActiveD2, const uint *edgeListOverloadD,
                           const uint *activeOverloadNodePointersD, bool *finished);

__global__ void
cc_kernelDynamicAsync(uint overloadStartNode, uint overloadNodeNum, const uint *overloadNodeListD, const uint *degreeD,
                      uint *valueD, uint *labelD1, uint *labelD2, const uint *edgeListOverloadD,
                      const uint *activeOverloadNodePointersD, bool *finished);

template <typename T>
__global__ void
cc_kernelStaticSwap(uint activeNodesNum, uint *activeNodeListD,
                    uint *staticNodePointerD, uint *degreeD,
                    uint *edgeListD, uint *valueD, T *isActiveD, bool *isInStaticD);

__global__ void
cc_kernelStaticAsync(uint activeNodesNum, const uint *activeNodeListD,
                     const uint *staticNodePointerD, const uint *degreeD,
                     const uint *edgeListD, uint *valueD, uint *labelD1, uint *labelD2, const bool *isInStaticD,
                     bool *finished, int *atomicValue);

__global__ void
bfs_kernelOptOfSorted(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD,
                      uint *edgeListOverload, uint *valueD, bool *labelD, bool *isInListD, uint *nodePointersOverloadD);

__global__ void
bfs_kernelShareOpt(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD,
                   uint *edgeListShare, uint *valueD, bool *labelD, uint overloadNode);

__global__ void
setLabelDefault(uint activeNum, uint *activeNodes, bool *labelD);

template<class T>
__global__ void
setLabelDefaultOpt(uint activeNum, uint *activeNodes, T *labelD);

__global__ void
mixStaticLabel(uint activeNum, uint *activeNodes, uint *labelD1, uint *labelD2, bool *isInD);

__global__ void
mixDynamicPartLabel(uint overloadPartNodeNum, uint startIndex, const uint *overloadNodes, uint *labelD1, uint *labelD2);

__global__ void
setDynamicPartLabelTrue(uint overloadPartNodeNum, uint startIndex, const uint *overloadNodes, uint *labelD1,
                        uint *labelD2);

__global__ void
mixCommonLabel(uint testNodeNum, uint *labelD1, uint *labelD2);

__global__ void
cleanStaticAndOverloadLabel(uint vertexNum, uint *staticLabel, uint *overloadLabel);

__global__ void
setStaticAndOverloadLabel(uint vertexNum, uint *activeLabel, uint *staticLabel, uint *overloadLabel, bool *isInD);

__global__ void
setStaticAndOverloadLabelBool(uint vertexNum, bool *activeLabel, bool *staticLabel, bool *overloadLabel, bool *isInD);

__global__ void
setStaticAndOverloadLabel4Pr(uint vertexNum, uint *activeLabel, uint *staticLabel, uint *overloadLabel, bool *isInD,
                             uint *fragmentRecordD, uint *nodePointersD, uint fragment_size, uint *degreeD,
                             bool *isFragmentActiveD);

__global__ void
setOverloadActiveNodeArray(uint vertexNum, uint *activeNodes, uint *overloadLabel,
                           uint *activeLabelPrefix);
template <typename T>
__global__ void
setStaticActiveNodeArray(uint vertexNum, uint *activeNodes, T *staticLabel,
                         uint *activeLabelPrefix);

__global__ void
cc_kernelStaticSwapOpt(uint activeNodesNum, uint *activeNodeListD,
                       uint *staticNodePointerD, uint *degreeD,
                       uint *edgeListD, uint *valueD, uint *isActiveD);

__global__ void
cc_kernelStaticSwapOpt2Label(uint activeNodesNum, uint *activeNodeListD,
                             uint *staticNodePointerD, uint *degreeD,
                             uint *edgeListD, uint *valueD, uint *isActiveD1, uint *isActiveD2, bool *isFinish);

__global__ void
setLabeling(uint vertexNum, bool *labelD, uint *labelingD);

__global__ void
setActiveNodeArray(uint vertexNum, uint *activeNodes, bool *activeLabel, uint *activeLabelPrefix);

__global__ void
setActiveNodeArrayAndNodePointer(uint vertexNum, uint *activeNodes, uint *activeNodePointers, bool *activeLabel,
                                 uint *activeLabelPrefix, uint overloadVertex, uint *degreeD);

__global__ void
setActiveNodeArrayAndNodePointerBySortOpt(uint vertexNum, uint *activeNodes, uint *activeOverloadDegree,
                                          bool *activeLabel, uint *activeLabelPrefix, bool *isInList, uint *degreeD);

__global__ void
setActiveNodeArrayAndNodePointerOpt(uint vertexNum, uint *activeNodes, uint *activeNodePointers, uint *activeLabel,
                                    uint *activeLabelPrefix, uint overloadVertex, uint *degreeD);

__global__ void
setActiveNodeArrayAndNodePointerSwap(uint vertexNum, uint *activeNodes, uint *activeLabel,
                                     uint *activeLabelPrefix, bool *isInD);

template <class T, typename E>
__global__ void
setOverloadNodePointerSwap(uint vertexNum, uint *activeNodes, E *activeNodePointers, T *activeLabel,
                           uint *activeLabelPrefix, uint *degreeD);

__global__ void
setFragmentData(uint activeNodeNum, uint *activeNodeList, uint *staticNodePointers, uint *staticFragmentData,
                uint staticFragmentNum, uint fragmentSize, bool *isInStatic);

__global__ void
setStaticFragmentData(uint staticFragmentNum, uint *canSwapFragmentD, uint *canSwapFragmentPrefixD,
                      uint *staticFragmentDataD);

__global__ void
setFragmentDataOpt(uint *staticFragmentData, uint staticFragmentNum, uint *staticFragmentVisitRecordsD);

__global__ void
recordFragmentVisit(uint *activeNodeListD, uint activeNodeNum, uint *nodePointersD, uint *degreeD, uint fragment_size,
                    uint *fragmentRecordsD);

__global__ void
bfsKernel_CommonPartition(uint startVertex, uint endVertex, uint offset, const bool *isActiveNodeListD,
                          const uint *nodePointersD,
                          const uint *edgeListD, const uint *degreeD, uint *valueD, bool *nextActiveNodeListD);

__global__ void
prSumKernel_CommonPartition(uint startVertex, uint endVertex, uint offset, const bool *isActiveNodeListD,
                            const uint *nodePointersD,
                            const uint *edgeListD, const uint *degreeD, const uint *outDegreeD, const float *valueD,
                            float *sumD);

__global__ void
prKernel_CommonPartition(uint nodeNum, float *valueD, float *sumD, bool *isActiveNodeList);

__global__ void
prSumKernel_UVM(uint vertexNum, const int *isActiveNodeListD, const uint *nodePointersD,
                const uint *edgeListD, const uint *degreeD, const float *valueD, float *sumD);

__global__ void
prKernel_UVM(uint nodeNum, float *valueD, float *sumD, int *isActiveListD);

__global__ void
prSumKernel_UVM_Out(uint vertexNum, int *isActiveNodeListD, const uint *nodePointersD,
                    const uint *edgeListD, const uint *degreeD, const uint *outDegreeD, float *valueD);

__global__ void
prKernel_UVM_outDegree(uint nodeNum, float *valueD, float *sumD, int *isActiveListD);

template<typename T>
__global__ void
prSumKernel_static(uint activeNum, const uint *activeNodeList,
                   const uint *nodePointersD,
                   const uint *edgeListD, const uint *degreeD, const uint *outDegreeD, const T *valueD,
                   T *sumD);

template <typename E, typename K>
__global__ void
prSumKernel_dynamic(uint overloadStartNode, uint overloadNodeNum, const uint *overloadNodeListD,
                    const E *nodePointersD,
                    const uint *edgeListD, const uint *degreeD, const uint *outDegreeD, const K *valueD,
                    K *sumD);

template <typename T, typename K>
__global__ void prKernel_Opt(uint nodeNum, K *valueD, K *sumD, T *isActiveNodeList);

__global__ void
setFragmentDataOpt4Pr(uint *staticFragmentData, uint fragmentNum, uint *fragmentVisitRecordsD,
                      bool *isActiveFragmentD, uint* fragmentNormalMap2StaticD, uint maxStaticFragment);

__global__ void
ccKernel_CommonPartition(uint startVertex, uint endVertex, uint offset, const bool *isActiveNodeListD,
                         const uint *nodePointersD,
                         const uint *edgeListD, const uint *degreeD, uint *valueD, bool *nextActiveNodeListD);

__global__ void
ssspKernel_CommonPartition(uint startVertex, uint endVertex, uint offset, const bool *isActiveNodeListD,
                           const uint *nodePointersD,
                           const EdgeWithWeight *edgeListD, const uint *degreeD, uint *valueD,
                           bool *nextActiveNodeListD);
__global__ void
setStaticAndOverloadLabelAndRecord(uint vertexNum, uint *activeLabel, uint *staticLabel, uint *overloadLabel,
                                   bool *isInD, uint *vertexVisitRecordD);

template<int NT>
__device__ int reduceInWarp(int idInWarp, bool data);


template<typename T, typename E>
__global__ void
bfs_kernelDynamicPart(uint overloadStartNode, uint overloadNodeNum, const uint *overloadNodeListD,
                      const uint *degreeD,
                      uint *valueD,
                      T *isActiveD, const uint *edgeListOverloadD,
                      const E *activeOverloadNodePointersD) {
    streamVertices(overloadNodeNum, [&](uint index) {
        uint traverseIndex = overloadStartNode + index;
        uint id = overloadNodeListD[traverseIndex];
        uint sourceValue = valueD[id];
        uint finalValue = sourceValue + 1;
        for (uint i = 0; i < degreeD[id]; i++) {
            uint vertexId = edgeListOverloadD[activeOverloadNodePointersD[traverseIndex] -
                                              activeOverloadNodePointersD[overloadStartNode] + i];

            //printf("source node %d dest node %d set true finalValue %d valueD[vertexId] %d\n", id, vertexId, finalValue, valueD[vertexId]);

            if (finalValue < valueD[vertexId]) {
                //printf("source node %d dest node %d set true finalValue %d valueD[vertexId] %d\n", id, vertexId, finalValue, valueD[vertexId]);
                isActiveD[vertexId] = 1;
                atomicMin(&valueD[vertexId], finalValue);
            } else {
                //printf("source node %d dest node %d set false\n", id, vertexId);
                //isActiveD[vertexId] = 0;
            }
        }
    });
}


template<class T>
__global__ void
bfs_kernelStatic(uint nodeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD, uint *valueD,
                 T *labelD) {
    streamVertices(nodeNum, [&](uint index) {
        uint id = activeNodesD[index];
        uint edgeIndex = nodePointersD[id];
        uint sourceValue = valueD[id];
        uint finalValue;
        //printf("degreeD[0] is %d\n", degreeD[id]);
        for (uint i = 0; i < degreeD[id]; i++) {
            finalValue = sourceValue + 1;
            uint vertexId;
            vertexId = edgeListD[edgeIndex + i];
            if (finalValue < valueD[vertexId]) {
                atomicMin(&valueD[vertexId], finalValue);
                labelD[vertexId] = 1;
            }
        }
    });
}


template<class T>
__global__ void
setLabelDefaultOpt(uint activeNum, uint *activeNodes, T *labelD) {
    streamVertices(activeNum, [&](uint vertexId) {
        if (labelD[activeNodes[vertexId]]) {
            labelD[activeNodes[vertexId]] = 0;
            //printf("vertex%d index %d true to %d \n", vertexId, activeNodes[vertexId], labelD[activeNodes[vertexId]]);
        }
    });
}


template <class T, typename E>
__global__ void
setOverloadNodePointerSwap(uint vertexNum, uint *activeNodes, E *activeNodePointers, T *activeLabel,
                           uint *activeLabelPrefix, uint *degreeD) {
    streamVertices(vertexNum, [&](uint vertexId) {
        if (activeLabel[vertexId]) {
            activeNodes[activeLabelPrefix[vertexId]] = vertexId;
            activeNodePointers[activeLabelPrefix[vertexId]] = degreeD[vertexId];
            activeLabel[vertexId] = 0;
            /*if (vertexId == 1) {
                printf("activeLabel %d activeLabelPrefix[vertexId] %d degreeD[vertexId] %d\n", activeLabel[vertexId], activeLabelPrefix[vertexId], activeNodePointers[activeLabelPrefix[vertexId]]);
            }*/
        }
    });
}


template <typename T>
__global__ void
setStaticActiveNodeArray(uint vertexNum, uint *activeNodes, T *staticLabel,
                         uint *activeLabelPrefix) {
    streamVertices(vertexNum, [&](uint vertexId) {
        if (staticLabel[vertexId]) {
            activeNodes[activeLabelPrefix[vertexId]] = vertexId;
            staticLabel[vertexId] = 0;
        }
    });
}


template <typename T>
__global__ void
cc_kernelStaticSwap(uint activeNodesNum, uint *activeNodeListD,
                    uint *staticNodePointerD, uint *degreeD,
                    uint *edgeListD, uint *valueD, T *isActiveD, bool *isInStaticD) {
    streamVertices(activeNodesNum, [&](uint index) {
        uint id = activeNodeListD[index];
        if (isInStaticD[id]) {
            uint edgeIndex = staticNodePointerD[id];
            uint sourceValue = valueD[id];
            for (uint i = 0; i < degreeD[id]; i++) {
                uint vertexId = edgeListD[edgeIndex + i];
                if (sourceValue < valueD[vertexId]) {
                    atomicMin(&valueD[vertexId], sourceValue);
                    isActiveD[vertexId] = 1;
                }
            }
        }
    });
}

template <typename T, typename E>
__global__ void
cc_kernelDynamicSwap(uint overloadStartNode, uint overloadNodeNum, const uint *overloadNodeListD, const uint *degreeD,
                     uint *valueD,
                     T *isActiveD, const uint *edgeListOverloadD,
                     const E *activeOverloadNodePointersD) {
    streamVertices(overloadNodeNum, [&](uint index) {
        uint traverseIndex = overloadStartNode + index;
        uint id = overloadNodeListD[traverseIndex];
        uint sourceValue = valueD[id];
        for (uint i = 0; i < degreeD[id]; i++) {
            uint vertexId = edgeListOverloadD[activeOverloadNodePointersD[traverseIndex] -
                                              activeOverloadNodePointersD[overloadStartNode] + i];
            if (sourceValue < valueD[vertexId]) {
                atomicMin(&valueD[vertexId], sourceValue);
                isActiveD[vertexId] = 1;
            }
        }
    });
}

template <typename T>
__global__ void
sssp_kernel(uint activeNum, const uint *activeNodesD, const uint *nodePointersD, const uint *degreeD, EdgeWithWeight *edgeListD,
            uint *valueD,
            T *labelD) {
    streamVertices(activeNum, [&](uint index) {
        uint id = activeNodesD[index];
        uint edgeIndex = nodePointersD[id];
        //printf("source vertex %d, edgeIndex is %d degree %d \n", id, edgeIndex, degreeD[id]);
        uint sourceValue = valueD[id];
        uint finalValue;

        for (uint i = edgeIndex; i < edgeIndex + degreeD[id]; i++) {
            finalValue = sourceValue + edgeListD[i].weight;
            uint vertexId = edgeListD[i].toNode;
            //printf("source vertex %d, edgeindex is %d destnode is %d \n", id, i, edgeListD[i].toNode);
            if (finalValue < valueD[vertexId]) {
                atomicMin(&valueD[vertexId], finalValue);
                labelD[vertexId] = 1;
            }
        }
    });
}

template <typename T, typename E>
__global__ void
sssp_kernelDynamic(uint overloadStartNode, uint overloadNodeNum, const uint *overloadNodeListD,
                   const uint *degreeD,
                   uint *valueD,
                   T *isActiveD, const EdgeWithWeight *edgeListOverloadD,
                   const E *activeOverloadNodePointersD) {
    streamVertices(overloadNodeNum, [&](uint index) {
        uint traverseIndex = overloadStartNode + index;
        uint id = overloadNodeListD[traverseIndex];
        uint sourceValue = valueD[id];
        uint finalValue;
        for (uint i = 0; i < degreeD[id]; i++) {
            EdgeWithWeight checkNode{};
            checkNode = edgeListOverloadD[activeOverloadNodePointersD[traverseIndex] -
                                          activeOverloadNodePointersD[overloadStartNode] + i];
            finalValue = sourceValue + checkNode.weight;
            uint vertexId = checkNode.toNode;
            if (finalValue < valueD[vertexId]) {
                //printf("source node %d dest node %d set true\n", id, vertexId);
                atomicMin(&valueD[vertexId], finalValue);
                isActiveD[vertexId] = 1;
            }
        }
    });
}
template <typename T, typename K>
__global__ void
prKernel_Opt(uint nodeNum, K *valueD, K *sumD, T *isActiveNodeList) {
    streamVertices(nodeNum, [&](uint index) {
        if (isActiveNodeList[index]) {
            K tempValue = 0.15 + 0.85 * sumD[index];
            K diff = tempValue > valueD[index] ? (tempValue - valueD[index]) : (valueD[index] - tempValue);

            if (diff > 0.001) {
                isActiveNodeList[index] = 1;
                valueD[index] = tempValue;
                sumD[index] = 0;
            } else {
                isActiveNodeList[index] = 0;
                sumD[index] = 0;
            }
        }

    });
}

template <typename E, typename K>
__global__ void
prSumKernel_dynamic(uint overloadStartNode, uint overloadNodeNum, const uint *overloadNodeListD,
                    const E *nodePointersD,
                    const uint *edgeListD, const uint *degreeD, const uint *outDegreeD, const K *valueD,
                    K *sumD) {
    streamVertices(overloadNodeNum, [&](uint index) {
        uint traverseIndex = overloadStartNode + index;
        uint nodeIndex = overloadNodeListD[traverseIndex];

        uint edgeOffset = nodePointersD[traverseIndex] - nodePointersD[overloadStartNode];
        K tempSum = 0;
        for (uint i = edgeOffset; i < edgeOffset + degreeD[nodeIndex]; i++) {
            uint srcNodeIndex = edgeListD[i];
            K tempValue = valueD[srcNodeIndex] / outDegreeD[srcNodeIndex];
            //printf("src %d dest %d value %f \n", srcNodeIndex,nodeIndex )
            tempSum += tempValue;
        }
        sumD[nodeIndex] = tempSum;

    });
}

template<typename T>
__global__ void
prSumKernel_static(uint activeNum, const uint *activeNodeList,
                   const uint *nodePointersD,
                   const uint *edgeListD, const uint *degreeD, const uint *outDegreeD, const T *valueD,
                   T *sumD) {
    streamVertices(activeNum, [&](uint index) {
        uint nodeIndex = activeNodeList[index];
        uint edgeIndex = nodePointersD[nodeIndex];
        T tempSum = 0;
        for (uint i = edgeIndex; i < edgeIndex + degreeD[nodeIndex]; i++) {
            uint srcNodeIndex = edgeListD[i];
            T tempValue = valueD[srcNodeIndex] / outDegreeD[srcNodeIndex];
            tempSum += tempValue;
        }
        sumD[nodeIndex] = tempSum;
    });
}

#endif