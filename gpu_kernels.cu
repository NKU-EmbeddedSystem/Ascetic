#include "gpu_kernels.cuh"




__device__ void gpu_sync(int goalVal, volatile int *arrayIn, volatile int *arrayOut) {
    int tidInBlock = blockDim.y * threadIdx.x + threadIdx.y;
    int blockId = blockIdx.x * gridDim.y + blockIdx.y;
    int nBlockNum = gridDim.x * gridDim.y;
    if (tidInBlock == 0) {
        arrayIn[blockId] = goalVal;
    }
    if (blockId == 0) {
        if (tidInBlock < nBlockNum) {
            while (arrayIn[tidInBlock] != goalVal) {
            }
        }
        __syncthreads();
        if (tidInBlock < nBlockNum) {
            arrayOut[tidInBlock] = goalVal;
        }
    }
    if (tidInBlock == 0) {
        while (arrayOut[blockId] != goalVal) {
        }
        if (blockId == 0) {
        }
    }
    __syncthreads();
}
//__device__ vola int g_mutex;
__device__ void gpu_sync(int goalVal, volatile int *g_mutex) {
    int tidInBlock = blockDim.y * threadIdx.x + threadIdx.y;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidInBlock == 0) {
        atomicAdd((int *) g_mutex, 1);
        while ((*g_mutex) != goalVal) {

        }
    }
    __syncthreads();
}

__global__ void
setLabeling(uint vertexNum, bool *labelD, uint *labelingD) {
    streamVertices(vertexNum, [&](uint vertexId) {
        if (labelD[vertexId]) {
            labelingD[vertexId] = 1;
            //printf("vertex[%d] set 1\n", vertexId);
        } else {
            labelingD[vertexId] = 0;
        }
    });
}

__global__ void
setActiveNodeArrayAndNodePointerOpt(uint vertexNum, uint *activeNodes, uint *activeNodePointers, uint *activeLabel,
                                    uint *activeLabelPrefix, uint overloadVertex, uint *degreeD) {
    streamVertices(vertexNum, [&](uint vertexId) {
        if (activeLabel[vertexId]) {
            activeNodes[activeLabelPrefix[vertexId]] = vertexId;
            if (vertexId > overloadVertex) {
                activeNodePointers[activeLabelPrefix[vertexId]] = degreeD[vertexId];
                activeLabel[vertexId] = 1;
            } else {
                activeNodePointers[activeLabelPrefix[vertexId]] = 0;
                activeLabel[vertexId] = 0;
            }
        }
    });
}

__global__ void
setActiveNodeArrayAndNodePointerSwap(uint vertexNum, uint *activeNodes, uint *activeLabel,
                                     uint *activeLabelPrefix, bool *isInD) {
    streamVertices(vertexNum, [&](uint vertexId) {
        if (activeLabel[vertexId]) {
            activeNodes[activeLabelPrefix[vertexId]] = vertexId;
            if (!isInD[vertexId]) {
                activeLabel[vertexId] = 1;
            } else {
                activeLabel[vertexId] = 0;
            }
        }
    });
}

__global__ void
setOverloadActiveNodeArray(uint vertexNum, uint *activeNodes, uint *overloadLabel,
                           uint *activeLabelPrefix) {
    streamVertices(vertexNum, [&](uint vertexId) {
        if (overloadLabel[vertexId]) {
            activeNodes[activeLabelPrefix[vertexId]] = vertexId;
        }
    });
}

__global__ void
setStaticAndOverloadLabel(uint vertexNum, uint *activeLabel, uint *staticLabel, uint *overloadLabel, bool *isInD) {
    streamVertices(vertexNum, [&](uint vertexId) {
        if (activeLabel[vertexId]) {
            if (isInD[vertexId]) {
                staticLabel[vertexId] = 1;
            } else {
                overloadLabel[vertexId] = 1;
            }
        }
    });
}

__global__ void
setStaticAndOverloadLabelBool(uint vertexNum, bool *activeLabel, bool *staticLabel, bool *overloadLabel, bool *isInD) {
    streamVertices(vertexNum, [&](uint vertexId) {
        if (activeLabel[vertexId]) {
            if (isInD[vertexId]) {
                staticLabel[vertexId] = 1;
            } else {
                overloadLabel[vertexId] = 1;
            }
        }
    });
}

__global__ void
setStaticAndOverloadLabelAndRecord(uint vertexNum, uint *activeLabel, uint *staticLabel, uint *overloadLabel,
                                   bool *isInD, uint *vertexVisitRecordD) {
    streamVertices(vertexNum, [&](uint vertexId) {
        if (activeLabel[vertexId]) {
            if (isInD[vertexId]) {
                staticLabel[vertexId] = 1;
            } else {
                overloadLabel[vertexId] = 1;
            }
            atomicAdd(&vertexVisitRecordD[vertexId], 1);
        }
    });
}

__global__ void
setStaticAndOverloadLabel4Pr(uint vertexNum, uint *activeLabel, uint *staticLabel, uint *overloadLabel, bool *isInD,
                             uint *fragmentRecordD, uint *nodePointersD, uint fragment_size, uint *degreeD,
                             bool *isFragmentActiveD) {
    streamVertices(vertexNum, [&](uint vertexId) {
        if (activeLabel[vertexId]) {
            if (isInD[vertexId]) {
                staticLabel[vertexId] = 1;
            } else {
                overloadLabel[vertexId] = 1;
            }
        } else {
            uint edgeIndex = nodePointersD[vertexId];
            uint fragmentIndex = edgeIndex / fragment_size;
            uint fragmentMoreIndex = (edgeIndex + degreeD[vertexId]) / fragment_size;
            if (isFragmentActiveD[fragmentIndex]) {
                if (fragmentMoreIndex > fragmentIndex) {
                    atomicAdd(&fragmentRecordD[fragmentIndex],
                              fragmentIndex * fragment_size + fragment_size - edgeIndex);
                } else {
                    atomicAdd(&fragmentRecordD[fragmentIndex], degreeD[vertexId]);
                }
            }
        }
    });
}

__global__ void
cleanStaticAndOverloadLabel(uint vertexNum, uint *staticLabel, uint *overloadLabel) {
    streamVertices(vertexNum, [&](uint vertexId) {
        staticLabel[vertexId] = 0;
        overloadLabel[vertexId] = 0;
    });
}

__global__ void
setStaticAndOverloadNodePointer(uint vertexNum, uint *staticNodes, uint *overloadNodes, uint *overloadNodeDegrees,
                                uint *staticLabel, uint *overloadLabel,
                                uint *staticPrefix, uint *overloadPrefix, uint *degreeD) {
    streamVertices(vertexNum, [&](uint vertexId) {
        if (overloadLabel[vertexId]) {
            overloadNodes[overloadPrefix[vertexId]] = vertexId;
            overloadNodeDegrees[overloadPrefix[vertexId]] = degreeD[vertexId];
            overloadLabel[vertexId] = 0;
        }
        if (staticLabel[vertexId]) {
            staticNodes[staticPrefix[vertexId]] = vertexId;
            staticLabel[vertexId] = 0;
        }
    });
}

__global__ void
setActiveNodeArray(uint vertexNum, uint *activeNodes, bool *activeLabel,
                   uint *activeLabelPrefix) {
    streamVertices(vertexNum, [&](uint vertexId) {
        if (activeLabel[vertexId]) {
            //printf("activeNodes %d set %d %d \n", activeLabelPrefix[vertexId], vertexId, activeLabel[vertexId]);
            activeNodes[activeLabelPrefix[vertexId]] = vertexId;
        }
    });
}

__global__ void
setActiveNodeArrayAndNodePointer(uint vertexNum, uint *activeNodes, uint *activeNodePointers, bool *activeLabel,
                                 uint *activeLabelPrefix, uint overloadVertex, uint *degreeD) {
    streamVertices(vertexNum, [&](uint vertexId) {
        if (activeLabel[vertexId]) {
            activeNodes[activeLabelPrefix[vertexId]] = vertexId;
            if (vertexId > overloadVertex) {
                activeNodePointers[activeLabelPrefix[vertexId]] = degreeD[vertexId];
            } else {
                activeNodePointers[activeLabelPrefix[vertexId]] = 0;
            }
        }
    });
}

__global__ void
setActiveNodeArrayAndNodePointerBySortOpt(uint vertexNum, uint *activeNodes, uint *activeOverloadDegree,
                                          bool *activeLabel, uint *activeLabelPrefix, bool *isInList, uint *degreeD) {
    streamVertices(vertexNum, [&](uint vertexId) {
        if (activeLabel[vertexId]) {
            activeNodes[activeLabelPrefix[vertexId]] = vertexId;
            if (!isInList[vertexId]) {
                activeOverloadDegree[activeLabelPrefix[vertexId]] = degreeD[vertexId];
            } else {
                activeOverloadDegree[activeLabelPrefix[vertexId]] = 0;
            }
        }
    });
}

__global__ void
setLabelDefault(uint activeNum, uint *activeNodes, bool *labelD) {
    streamVertices(activeNum, [&](uint vertexId) {
        if (labelD[activeNodes[vertexId]]) {
            labelD[activeNodes[vertexId]] = 0;
            //printf("vertex%d index %d true to %d \n", vertexId, activeNodes[vertexId], labelD[activeNodes[vertexId]]);
        }
    });
}

__global__ void
mixStaticLabel(uint activeNum, uint *activeNodes, uint *labelD1, uint *labelD2, bool *isInD) {
    streamVertices(activeNum, [&](uint index) {
        uint vertexId = activeNodes[index];
        if (labelD1[vertexId]) {
            labelD1[vertexId] = 0;
        }
        if (isInD[vertexId]) {
            labelD1[vertexId] = 1;
        }
        labelD2[vertexId] = 0;
    });
}

__global__ void
mixDynamicPartLabel(uint overloadPartNodeNum, uint startIndex, const uint *overloadNodes, uint *labelD1,
                    uint *labelD2) {
    streamVertices(overloadPartNodeNum, [&](uint index) {
        uint vertexId = overloadNodes[startIndex + index];
        labelD1[vertexId] = labelD1[vertexId] || labelD2[vertexId];
        labelD2[vertexId] = 0;
    });
}

__global__ void
mixCommonLabel(uint testNodeNum, uint *labelD1, uint *labelD2) {
    streamVertices(testNodeNum, [&](uint vertexId) {
        labelD1[vertexId] = labelD1[vertexId] || labelD2[vertexId];
        labelD2[vertexId] = 0;
    });
}

__global__ void
setDynamicPartLabelTrue(uint overloadPartNodeNum, uint startIndex, const uint *overloadNodes, uint *labelD1,
                        uint *labelD2) {
    streamVertices(overloadPartNodeNum, [&](uint index) {
        uint vertexId = overloadNodes[startIndex + index];
        labelD1[vertexId] = true;
        labelD2[vertexId] = false;
    });
}

__global__ void
bfs_kernel(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD, uint *valueD,
           bool *labelD) {
    streamVertices(activeNum, [&](uint index) {
        uint id = activeNodesD[index];
        uint edgeIndex = nodePointersD[id];
        uint sourceValue = valueD[id];
        uint finalValue;
        for (uint i = edgeIndex; i < edgeIndex + degreeD[id]; i++) {
            finalValue = sourceValue + 1;
            if (finalValue < valueD[edgeListD[i]]) {
                atomicMin(&valueD[edgeListD[i]], finalValue);
                labelD[edgeListD[i]] = true;
            }
        }
    });
}

__global__ void
bfsKernel_CommonPartition(uint startVertex, uint endVertex, uint offset, const bool *isActiveNodeListD,
                          const uint *nodePointersD,
                          const uint *edgeListD, const uint *degreeD, uint *valueD, bool *nextActiveNodeListD) {
    streamVertices(endVertex - startVertex + 1, [&](uint index) {
        uint nodeIndex = startVertex + index;
        if (isActiveNodeListD[nodeIndex]) {
            uint edgeIndex = nodePointersD[nodeIndex] - offset;
            uint sourceValue = valueD[nodeIndex];
            uint finalValue;
            //printf("node %d edgeIndex %d sourceValue %d degreeD[nodeIndex] %d\n", nodeIndex, edgeIndex, sourceValue, degreeD[nodeIndex]);
            for (uint i = edgeIndex; i < edgeIndex + degreeD[nodeIndex]; i++) {

                //printf("node %d dest node %d set true \n", nodeIndex, edgeListD[i]);
                finalValue = sourceValue + 1;
                if (finalValue < valueD[edgeListD[i]]) {
                    atomicMin(&valueD[edgeListD[i]], finalValue);
                    nextActiveNodeListD[edgeListD[i]] = true;
                }
            }
        }
    });
}

__global__ void
prSumKernel_CommonPartition(uint startVertex, uint endVertex, uint offset, const bool *isActiveNodeListD,
                            const uint *nodePointersD,
                            const uint *edgeListD, const uint *degreeD, const uint *outDegreeD, const float *valueD,
                            float *sumD) {
    streamVertices(endVertex - startVertex + 1, [&](uint index) {
        uint nodeIndex = startVertex + index;
        if (isActiveNodeListD[nodeIndex]) {
            uint edgeIndex = nodePointersD[nodeIndex] - offset;
            float tempSum = 0;
            for (uint i = edgeIndex; i < edgeIndex + degreeD[nodeIndex]; i++) {
                uint srcNodeIndex = edgeListD[i];
                float tempValue = valueD[srcNodeIndex] / outDegreeD[srcNodeIndex];
                tempSum += tempValue;
            }
            sumD[nodeIndex] = tempSum;
        }
    });
}

__global__ void
prKernel_CommonPartition(uint nodeNum, float *valueD, float *sumD, bool *isActiveNodeList) {
    streamVertices(nodeNum, [&](uint index) {
        if (isActiveNodeList[index]) {
            float tempValue = 0.15 + 0.85 * sumD[index];
            float diff = tempValue > valueD[index] ? (tempValue - valueD[index]) : (valueD[index] - tempValue);
            /*if (index == 1) {
                printf("tempValue %f \n", tempValue);
            }*/
            if (diff > 0.001) {
                isActiveNodeList[index] = true;
                valueD[index] = tempValue;
                sumD[index] = 0;
            } else {
                isActiveNodeList[index] = false;
                sumD[index] = 0;
            }
        }

    });
}


__global__ void
prSumKernel_UVM(uint vertexNum, const int *isActiveNodeListD, const uint *nodePointersD,
                const uint *edgeListD, const uint *degreeD, const float *valueD, float *sumD) {
    streamVertices(vertexNum, [&](uint index) {
        uint nodeIndex = index;
        if (isActiveNodeListD[nodeIndex] > 0) {
            uint edgeIndex = nodePointersD[nodeIndex];
            float sourceValue = valueD[nodeIndex] / degreeD[nodeIndex];
            //printf("node %d edgeIndex %d sourceValue %d degreeD[nodeIndex] %d\n", nodeIndex, edgeIndex, sourceValue, degreeD[nodeIndex]);
            for (uint i = edgeIndex; i < edgeIndex + degreeD[nodeIndex]; i++) {

                //printf("node %d dest node %d set true \n", nodeIndex, edgeListD[i]);
                atomicAdd(&sumD[edgeListD[i]], sourceValue);
            }
        }
    });
}

__global__ void
prSumKernel_UVM_Out(uint vertexNum, int *isActiveNodeListD, const uint *nodePointersD,
                    const uint *edgeListD, const uint *degreeD, const uint *outDegreeD, float *valueD) {
    streamVertices(vertexNum, [&](uint index) {
        uint nodeIndex = index;
        if (isActiveNodeListD[nodeIndex] > 0) {
            uint edgeIndex = nodePointersD[nodeIndex];
            float tempSum = 0;
            for (uint i = edgeIndex; i < edgeIndex + degreeD[nodeIndex]; i++) {
                uint srcNodeIndex = edgeListD[i];
                float tempValue = valueD[srcNodeIndex] / outDegreeD[srcNodeIndex];
                tempSum += tempValue;
            }

            float tempValue = 0.15 + 0.85 * tempSum;
            float diff =
                    tempValue > valueD[nodeIndex] ? (tempValue - valueD[nodeIndex]) : (valueD[nodeIndex] - tempValue);
            if (diff > 0.0001) {
                isActiveNodeListD[nodeIndex] = 1;
                valueD[index] = tempValue;
                //sumD[index] = 0;
            } else {
                isActiveNodeListD[nodeIndex] = 0;
                valueD[index] = tempValue;
                //sumD[index] = 0;
            }

            if (index >= 0 && index <= 10) {
                printf("value %d is %f \n", index, valueD[index]);
            }
        }
    });
}

__global__ void
prKernel_UVM(uint nodeNum, float *valueD, float *sumD, int *isActiveListD) {
    streamVertices(nodeNum, [&](uint index) {
        float tempValue = 0.15 + 0.85 * sumD[index];
        float diff = tempValue > valueD[index] ? (tempValue - valueD[index]) : (valueD[index] - tempValue);
        if (diff > 0.001) {
            isActiveListD[index] = 1;
            valueD[index] = tempValue;
            sumD[index] = 0;
        } else {
            isActiveListD[index] = 1;
            valueD[index] = tempValue;
            sumD[index] = 0;
        }

        if (index >= 0 && index <= 10) {
            printf("value %d is %f \n", index, valueD[index]);
        }
    });
}

__global__ void
prKernel_UVM_outDegree(uint nodeNum, float *valueD, float *sumD, int *isActiveListD) {
    streamVertices(nodeNum, [&](uint index) {
        float tempValue = 0.15 + 0.85 * sumD[index];
        float diff = tempValue > valueD[index] ? (tempValue - valueD[index]) : (valueD[index] - tempValue);
        if (diff > 0.001) {
            isActiveListD[index] = 1;
            valueD[index] = tempValue;
            sumD[index] = 0;
        } else {
            isActiveListD[index] = 0;
            //valueD[index] = tempValue;
            sumD[index] = 0;
        }

        if (index >= 0 && index <= 10) {
            printf("value %d is %f \n", index, valueD[index]);
        }
    });
}

__global__ void
cc_kernel(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD, uint *valueD,
          bool *labelD) {
    streamVertices(activeNum, [&](uint index) {
        uint id = activeNodesD[index];
        uint edgeIndex = nodePointersD[id];
        uint sourceValue = valueD[id];
        for (uint i = edgeIndex; i < edgeIndex + degreeD[id]; i++) {
            if (sourceValue < valueD[edgeListD[i]]) {
                atomicMin(&valueD[edgeListD[i]], sourceValue);
                labelD[edgeListD[i]] = true;
            }
        }
    });
}

__global__ void
ccKernel_CommonPartition(uint startVertex, uint endVertex, uint offset, const bool *isActiveNodeListD,
                         const uint *nodePointersD,
                         const uint *edgeListD, const uint *degreeD, uint *valueD, bool *nextActiveNodeListD) {
    streamVertices(endVertex - startVertex + 1, [&](uint index) {
        uint nodeIndex = startVertex + index;
        if (isActiveNodeListD[nodeIndex]) {
            uint edgeIndex = nodePointersD[nodeIndex] - offset;
            uint sourceValue = valueD[nodeIndex];
            for (uint i = edgeIndex; i < edgeIndex + degreeD[nodeIndex]; i++) {
                if (sourceValue < valueD[edgeListD[i]]) {
                    atomicMin(&valueD[edgeListD[i]], sourceValue);
                    nextActiveNodeListD[edgeListD[i]] = true;
                }
            }
        }
    });
}

__global__ void
ssspKernel_CommonPartition(uint startVertex, uint endVertex, uint offset, const bool *isActiveNodeListD,
                           const uint *nodePointersD,
                           const EdgeWithWeight *edgeListD, const uint *degreeD, uint *valueD,
                           bool *nextActiveNodeListD) {
    streamVertices(endVertex - startVertex + 1, [&](uint index) {
        uint nodeIndex = startVertex + index;
        if (isActiveNodeListD[nodeIndex]) {
            uint edgeIndex = nodePointersD[nodeIndex] - offset;
            uint sourceValue = valueD[nodeIndex];
            uint finalValue;
            for (uint i = edgeIndex; i < edgeIndex + degreeD[nodeIndex]; i++) {
                finalValue = sourceValue + edgeListD[i].weight;
                uint vertexId = edgeListD[i].toNode;
                if (finalValue < valueD[vertexId]) {
                    atomicMin(&valueD[vertexId], finalValue);
                    nextActiveNodeListD[vertexId] = true;
                }
            }
        }
    });
}

__global__ void
bfs_kernelShareOpt(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD,
                   uint *edgeListShare, uint *valueD, bool *labelD, uint overloadNode) {
    streamVertices(activeNum, [&](uint index) {
        uint id = activeNodesD[index];
        if (id >= overloadNode) {
            uint edgeIndex = nodePointersD[id];
            uint sourceValue = valueD[id];
            uint finalValue;
            for (uint i = edgeIndex; i < edgeIndex + degreeD[id]; i++) {
                finalValue = sourceValue + 1;
                if (finalValue < valueD[edgeListShare[i]]) {
                    atomicMin(&valueD[edgeListShare[i]], finalValue);
                    labelD[edgeListShare[i]] = true;
                    //printf("vertext[%d](edge[%d]) set 1\n", edgeListD[i], i);
                }
            }
        } else {
            uint edgeIndex = nodePointersD[id];
            uint sourceValue = valueD[id];
            uint finalValue;
            for (uint i = edgeIndex; i < edgeIndex + degreeD[id]; i++) {
                finalValue = sourceValue + 1;
                if (finalValue < valueD[edgeListD[i]]) {
                    atomicMin(&valueD[edgeListD[i]], finalValue);
                    labelD[edgeListD[i]] = true;
                    //printf("vertext[%d](edge[%d]) set 1\n", edgeListD[i], i);
                }
            }
        }

        //printf("index %d vertex %d edgeIndex %d degree %d sourcevalue %d \n", index, id, edgeIndex, degreeD[id], sourceValue);
    });
}

__global__ void
bfs_kernelStatic2Label(uint nodeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD,
                       uint *valueD,
                       uint *isActiveD1, uint *isActiveD2) {
    streamVertices(nodeNum, [&](uint index) {
        uint id = activeNodesD[index];
        if (isActiveD1[id]) {
            isActiveD1[id] = 0;
            uint edgeIndex = nodePointersD[id];
            uint sourceValue = valueD[id];
            uint finalValue;
            for (uint i = 0; i < degreeD[id]; i++) {
                finalValue = sourceValue + 1;
                uint vertexId;
                vertexId = edgeListD[edgeIndex + i];
                if (finalValue < valueD[vertexId]) {
                    atomicMin(&valueD[vertexId], finalValue);
                    isActiveD2[vertexId] = 1;
                }
            }
        }
    });
}


__global__ void
bfs_kernelDynamic2Label(uint overloadStartNode, uint overloadNodeNum, const uint *overloadNodeListD,
                        const uint *degreeD,
                        uint *valueD,
                        uint *isActiveD1, uint *isActiveD2, const uint *edgeListOverloadD,
                        const uint *activeOverloadNodePointersD) {
    streamVertices(overloadNodeNum, [&](uint index) {
        uint traverseIndex = overloadStartNode + index;
        uint id = overloadNodeListD[traverseIndex];
        uint sourceValue = valueD[id];
        uint finalValue = sourceValue + 1;
        if (isActiveD1[id]) {
            isActiveD1[id] = 0;
            for (uint i = 0; i < degreeD[id]; i++) {
                uint vertexId = edgeListOverloadD[activeOverloadNodePointersD[traverseIndex] -
                                                  activeOverloadNodePointersD[overloadStartNode] + i];
                if (finalValue < valueD[vertexId]) {
                    //printf("source node %d dest node %d set true\n", id, vertexId);
                    atomicMin(&valueD[vertexId], finalValue);
                    isActiveD2[vertexId] = 1;
                }
            }
        }
    });
}


__global__ void
sssp_kernelDynamicSwap2Label(uint overloadStartNode, uint overloadNodeNum, const uint *overloadNodeListD,
                             const uint *degreeD,
                             uint *valueD,
                             uint *isActiveD1, uint *isActiveD2, const EdgeWithWeight *edgeListOverloadD,
                             const uint *activeOverloadNodePointersD, bool *finished) {
    streamVertices(overloadNodeNum, [&](uint index) {
        uint traverseIndex = overloadStartNode + index;
        uint id = overloadNodeListD[traverseIndex];
        if (isActiveD1[id]) {
            isActiveD1[id] = 0;
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
                    isActiveD2[vertexId] = 1;
                    *finished = false;
                }
            }
        }
    });
}

__global__ void
sssp_kernelDynamicUvm(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, EdgeWithWeight *edgeListD,
                      uint *valueD,
                      uint *labelD1, uint *labelD2) {
    streamVertices(activeNum, [&](uint index) {
        uint id = activeNodesD[index];
        if (labelD1[id]) {
            labelD1[id] = 0;
        }
        uint edgeIndex = nodePointersD[index];
        /*if (isTest) {
            printf("index %d source vertex %d, edgeIndex is %d degree %d \n", index, id, edgeIndex, degreeD[id]);
        }*/
        uint sourceValue = valueD[id];
        uint finalValue;
        for (uint i = edgeIndex; i < edgeIndex + degreeD[id]; i++) {
            finalValue = sourceValue + edgeListD[i].weight;
            uint vertexId = edgeListD[i].toNode;
            //printf("source vertex %d, edgeindex is %d destnode is %d \n", id, i, edgeListD[i].toNode);
            if (finalValue < valueD[vertexId]) {
                atomicMin(&valueD[vertexId], finalValue);
                labelD2[vertexId] = 1;
            }
        }
    });
}


__global__ void
bfs_kernelStaticSwap(uint nodeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD,
                     uint *valueD,
                     uint *labelD, uint *fragmentRecordsD, uint fragment_size, uint maxpartionSize, uint testNumNodes) {
    streamVertices(nodeNum, [&](uint index) {
        uint id = activeNodesD[index];
        uint edgeIndex = nodePointersD[id];
        uint sourceValue = valueD[id];
        uint finalValue;

        uint fragmentIndex = edgeIndex / fragment_size;
        uint fragmentMoreIndex = (edgeIndex + degreeD[id]) / fragment_size;
        if (fragmentMoreIndex > fragmentIndex) {
            atomicAdd(&fragmentRecordsD[fragmentIndex], fragmentIndex * fragment_size + fragment_size - edgeIndex);
        } else {
            atomicAdd(&fragmentRecordsD[fragmentIndex], degreeD[id]);
        }

        for (uint i = 0; i < degreeD[id]; i++) {
            finalValue = sourceValue + 1;
            uint vertexId;
            if (finalValue < valueD[vertexId]) {
                atomicMin(&valueD[vertexId], finalValue);
                labelD[vertexId] = 1;
            }
        }
    });
}

__global__ void
bfs_kernelStaticSwap(uint nodeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD,
                     uint *valueD,
                     uint *labelD, bool *isInD, uint *fragmentRecordsD, uint fragment_size) {
    streamVertices(nodeNum, [&](uint index) {
        uint id = activeNodesD[index];
        if (isInD[id]) {
            uint edgeIndex = nodePointersD[id];
            uint sourceValue = valueD[id];
            uint finalValue;
            uint fragmentIndex = edgeIndex / fragment_size;
            uint fragmentMoreIndex = (edgeIndex + degreeD[id]) / fragment_size;
            if (fragmentMoreIndex > fragmentIndex) {
                atomicAdd(&fragmentRecordsD[fragmentIndex], fragmentIndex * fragment_size + fragment_size - edgeIndex);
            } else {
                atomicAdd(&fragmentRecordsD[fragmentIndex], degreeD[id]);
            }
            for (uint i = 0; i < degreeD[id]; i++) {
                finalValue = sourceValue + 1;
                uint vertexId;
                vertexId = edgeListD[edgeIndex + i];
                if (finalValue < valueD[vertexId]) {
                    atomicMin(&valueD[vertexId], finalValue);
                    labelD[vertexId] = 1;
                }
            }
        }
    });
}

__global__ void
recordFragmentVisit(uint *activeNodeListD, uint activeNodeNum, uint *nodePointersD, uint *degreeD, uint fragment_size,
                    uint *fragmentRecordsD) {
    streamVertices(activeNodeNum, [&](uint index) {
        uint id = activeNodeListD[index];
        uint edgeIndex = nodePointersD[id];
        uint fragmentIndex = edgeIndex / fragment_size;
        uint fragmentMoreIndex = (edgeIndex + degreeD[id]) / fragment_size;
        if (fragmentMoreIndex > fragmentIndex) {
            atomicAdd(&fragmentRecordsD[fragmentIndex], fragmentIndex * fragment_size + fragment_size - edgeIndex);
        } else {
            atomicAdd(&fragmentRecordsD[fragmentIndex], degreeD[id]);
        }
    });
}

/*__global__ void
bfs_kernelDynamic(uint activeNum, uint *activeNodesD, uint *degreeD, uint *valueD,
                  uint *labelD, uint overloadNode, uint *overloadEdgeListD,
                  uint *nodePointersOverloadD) {
    streamVertices(activeNum, [&](uint index) {
        uint id = activeNodesD[index];
        uint sourceValue = valueD[id];
        uint finalValue;
        if (id > overloadNode) {
            for (uint i = 0; i < degreeD[id]; i++) {
                finalValue = sourceValue + 1;
                uint vertexId = overloadEdgeListD[nodePointersOverloadD[index] + i];
                if (finalValue < valueD[vertexId]) {
                    atomicMin(&valueD[vertexId], finalValue);
                    labelD[vertexId] = 1;
                }
            }
        }
    });
}*/

__global__ void
bfs_kernelDynamic(uint activeNum, uint *activeNodesD, uint *degreeD, uint *valueD,
                  uint *labelD, uint overloadNode, uint *overloadEdgeListD,
                  uint *nodePointersOverloadD) {
    streamVertices(overloadNode, [&](uint index) {
        uint theIndex = activeNum - overloadNode + index;
        uint id = activeNodesD[theIndex];
        uint sourceValue = valueD[id];
        uint finalValue;
        for (uint i = 0; i < degreeD[id]; i++) {
            finalValue = sourceValue + 1;
            uint vertexId = overloadEdgeListD[nodePointersOverloadD[theIndex] + i];
            if (finalValue < valueD[vertexId]) {
                atomicMin(&valueD[vertexId], finalValue);
                labelD[vertexId] = 1;
            }
        }
    });
}

__global__ void
bfs_kernelDynamicSwap(uint activeNum, uint *activeNodesD, uint *degreeD, uint *valueD,
                      uint *labelD, uint *overloadEdgeListD,
                      uint *nodePointersOverloadD) {
    streamVertices(activeNum, [&](uint index) {
        uint id = activeNodesD[index];
        uint sourceValue = valueD[id];
        uint finalValue;
        for (uint i = 0; i < degreeD[id]; i++) {
            finalValue = sourceValue + 1;
            uint vertexId = overloadEdgeListD[nodePointersOverloadD[index] + i];
            if (finalValue < valueD[vertexId]) {
                atomicMin(&valueD[vertexId], finalValue);
                labelD[vertexId] = 1;
            }
        }
    });
}

__global__ void
bfs_kernelOpt(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD,
              uint *valueD,
              bool *labelD, uint overloadNode, uint *overloadEdgeListD, uint *nodePointersOverloadD) {
    streamVertices(activeNum, [&](uint index) {
        uint id = activeNodesD[index];

        uint edgeIndex = nodePointersD[id];
        uint sourceValue = valueD[id];
        uint finalValue;
        for (uint i = 0; i < degreeD[id]; i++) {
            finalValue = sourceValue + 1;
            uint vertexId;
            if (id > overloadNode) {
                vertexId = overloadEdgeListD[nodePointersOverloadD[index] + i];
            } else {
                vertexId = edgeListD[edgeIndex + i];
            }
            if (finalValue < valueD[vertexId]) {
                atomicMin(&valueD[vertexId], finalValue);
                labelD[vertexId] = true;
            }
        }
    });
}

__global__ void
cc_kernelOpt(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD, uint *valueD,
             bool *labelD, uint overloadNode, uint *overloadEdgeListD, uint *nodePointersOverloadD) {
    streamVertices(activeNum, [&](uint index) {
        uint id = activeNodesD[index];
        uint edgeIndex = nodePointersD[id];
        uint sourceValue = valueD[id];
        for (uint i = 0; i < degreeD[id]; i++) {
            uint vertexId;
            if (id > overloadNode) {
                vertexId = overloadEdgeListD[nodePointersOverloadD[index] + i];
            } else {
                vertexId = edgeListD[edgeIndex + i];
            }
            if (sourceValue < valueD[vertexId]) {
                atomicMin(&valueD[vertexId], sourceValue);
                labelD[vertexId] = true;
            }
        }
    });
}

__global__ void
cc_kernelStaticSwapOpt2Label(uint activeNodesNum, uint *activeNodeListD,
                             uint *staticNodePointerD, uint *degreeD,
                             uint *edgeListD, uint *valueD, uint *isActiveD1, uint *isActiveD2, bool *isFinish) {
    streamVertices(activeNodesNum, [&](uint index) {
        uint id = activeNodeListD[index];
        if (isActiveD1[id]) {
            isActiveD1[id] = 0;
            uint edgeIndex = staticNodePointerD[id];
            uint sourceValue = valueD[id];
            for (uint i = 0; i < degreeD[id]; i++) {
                uint vertexId = edgeListD[edgeIndex + i];
                if (sourceValue < valueD[vertexId]) {
                    atomicMin(&valueD[vertexId], sourceValue);
                    isActiveD2[vertexId] = 1;
                    *isFinish = false;
                }
            }
        }

    });
}

__global__ void
cc_kernelStaticSwapOpt(uint activeNodesNum, uint *activeNodeListD,
                       uint *staticNodePointerD, uint *degreeD,
                       uint *edgeListD, uint *valueD, uint *isActiveD) {
    streamVertices(activeNodesNum, [&](uint index) {
        uint id = activeNodeListD[index];
        uint edgeIndex = staticNodePointerD[id];
        uint sourceValue = valueD[id];
        for (uint i = 0; i < degreeD[id]; i++) {
            uint vertexId = edgeListD[edgeIndex + i];
            if (sourceValue < valueD[vertexId]) {
                atomicMin(&valueD[vertexId], sourceValue);
                isActiveD[vertexId] = 1;
            }
        }
    });
}


__global__ void
cc_kernelStaticAsync(uint activeNodesNum, const uint *activeNodeListD,
                     const uint *staticNodePointerD, const uint *degreeD,
                     const uint *edgeListD, uint *valueD, uint *labelD1, uint *labelD2, const bool *isInStaticD,
                     bool *finished, int *atomicValue) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int tidInBlock = blockDim.y * threadIdx.x + threadIdx.y;
    int blockId = blockIdx.x * gridDim.y + blockIdx.y;
    int iter = 0;
    uint *checkLabel = iter % 2 == 0 ? labelD1 : labelD2;
    uint *targetLabel = iter % 2 == 0 ? labelD2 : labelD1;
    int syncIndex = 1;
    volatile bool *testFinish = (bool *) finished;
    *testFinish = false;
    gpu_sync(gridDim.x, atomicValue);
    //__syncthreads();
    if (tidInBlock == 0) {
        printf("tid %d blockid %d testFinish %d \n", tid, blockId, *testFinish);
    }
    if (tid == 0) {
        *testFinish = false;
    }
    gpu_sync(2 * gridDim.x, atomicValue);
    if (*testFinish) {
        if (tidInBlock == 0) {
            printf("1 tid %d blockid %d testFinish %d \n", tid, blockId, *testFinish);
        }
        return;
    }
    if (tid == 0) {
        *testFinish = true;
    }
    gpu_sync(2 * gridDim.x, atomicValue);
    if (*testFinish) {
        if (tidInBlock == 0) {
            printf("2 tid %d blockid %d testFinish %d \n", tid, blockId, *testFinish);
        }
        return;
    }
    printf("================\n");
}

__global__ void
cc_kernelDynamicAsync(uint overloadStartNode, uint overloadNodeNum, const uint *overloadNodeListD, const uint *degreeD,
                      uint *valueD, uint *labelD1, uint *labelD2, const uint *edgeListOverloadD,
                      const uint *activeOverloadNodePointersD, bool *finished) {
    streamVertices(overloadNodeNum, [&](uint index) {
        uint traverseIndex = overloadStartNode + index;
        uint id = overloadNodeListD[traverseIndex];
        if (labelD1[id]) {
            labelD1[id] = 0;
            uint sourceValue = valueD[id];
            for (uint i = 0; i < degreeD[id]; i++) {
                uint vertexId = edgeListOverloadD[activeOverloadNodePointersD[traverseIndex] -
                                                  activeOverloadNodePointersD[overloadStartNode] + i];
                if (sourceValue < valueD[vertexId]) {
                    atomicMin(&valueD[vertexId], sourceValue);
                    *finished = false;
                    labelD2[vertexId] = 1;
                }
            }
        }
    });
}


__global__ void
cc_kernelDynamicSwap2Label(uint overloadStartNode, uint overloadNodeNum, const uint *overloadNodeListD,
                           const uint *degreeD,
                           uint *valueD,
                           uint *isActiveD1, uint *isActiveD2, const uint *edgeListOverloadD,
                           const uint *activeOverloadNodePointersD, bool *finished) {
    streamVertices(overloadNodeNum, [&](uint index) {
        uint traverseIndex = overloadStartNode + index;
        uint id = overloadNodeListD[traverseIndex];
        if (isActiveD1[id]) {
            isActiveD1[id] = 0;
            uint sourceValue = valueD[id];
            for (uint i = 0; i < degreeD[id]; i++) {
                uint vertexId = edgeListOverloadD[activeOverloadNodePointersD[traverseIndex] -
                                                  activeOverloadNodePointersD[overloadStartNode] + i];
                if (sourceValue < valueD[vertexId]) {
                    atomicMin(&valueD[vertexId], sourceValue);
                    isActiveD2[vertexId] = 1;
                    *finished = false;
                }
            }
        }
    });
}

__global__ void
sssp_kernelStaticSwapOpt2Label(uint activeNodesNum, const uint *activeNodeListD,
                               const uint *staticNodePointerD, const uint *degreeD,
                               EdgeWithWeight *edgeListD, uint *valueD, uint *isActiveD1, uint *isActiveD2,
                               bool *isFinish) {

    streamVertices(activeNodesNum, [&](uint index) {
        uint id = activeNodeListD[index];
        if (isActiveD1[id]) {
            isActiveD1[id] = 0;
            uint edgeIndex = staticNodePointerD[id];
            uint sourceValue = valueD[id];
            uint finalValue;
            for (uint i = 0; i < degreeD[id]; i++) {
                EdgeWithWeight checkNode{};
                checkNode = edgeListD[edgeIndex + i];
                finalValue = sourceValue + checkNode.weight;
                uint vertexId = checkNode.toNode;
                if (finalValue < valueD[vertexId]) {
                    atomicMin(&valueD[vertexId], finalValue);
                    isActiveD2[vertexId] = 1;
                    *isFinish = false;
                }
            }
        }

    });
}


__global__ void
sssp_kernelOpt(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, EdgeWithWeight *edgeListD,
               uint *valueD,
               bool *labelD, uint overloadNode, EdgeWithWeight *overloadEdgeListD, uint *nodePointersOverloadD) {
    streamVertices(activeNum, [&](uint index) {
        uint id = activeNodesD[index];
        uint edgeIndex = nodePointersD[id];
        uint sourceValue = valueD[id];
        uint finalValue;
        for (uint i = 0; i < degreeD[id]; i++) {
            EdgeWithWeight checkNode{};
            if (id > overloadNode) {
                checkNode = overloadEdgeListD[nodePointersOverloadD[index] + i];
            } else {
                checkNode = edgeListD[edgeIndex + i];
            }
            finalValue = sourceValue + checkNode.weight;
            uint vertexId = checkNode.toNode;
            if (finalValue < valueD[vertexId]) {
                atomicMin(&valueD[vertexId], finalValue);
                labelD[vertexId] = true;
                //printf("source vertex %d, toNode is %d \n", id, vertexId);
            }
        }
    });
}

__global__ void
bfs_kernelOptOfSorted(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD,
                      uint *edgeListOverload, uint *valueD, bool *labelD, bool *isInListD,
                      uint *nodePointersOverloadD) {
    streamVertices(activeNum, [&](uint index) {
        uint id = activeNodesD[index];
        uint sourceValue = valueD[id];
        uint finalValue;
        uint edgeIndex;
        uint *edgeList;
        if (!isInListD[id]) {
            edgeIndex = nodePointersOverloadD[index];
            edgeList = edgeListOverload;
        } else {
            edgeIndex = nodePointersD[id];
            edgeList = edgeListD;
        }

        for (uint i = 0; i < degreeD[id]; i++) {
            finalValue = sourceValue + 1;
            uint vertexId = edgeList[edgeIndex + i];
            if (finalValue < valueD[vertexId]) {
                atomicMin(&valueD[vertexId], finalValue);
                labelD[vertexId] = true;
                //printf("vertext[%d](edge[%d]) set 1\n", edgeListD[i], i);
            }
        }
    });
}

__global__ void
setFragmentData(uint activeNodeNum, uint *activeNodeList, uint *staticNodePointers, uint *staticFragmentData,
                uint staticFragmentNum, uint fragmentSize, bool *isInStatic) {
    streamVertices(activeNodeNum, [&](uint index) {
        uint vertexId = activeNodeList[index];
        if (isInStatic[vertexId]) {
            uint staticFragmentIndex = staticNodePointers[vertexId] / fragmentSize;
            if (staticFragmentIndex < staticFragmentNum) {
                staticFragmentData[staticFragmentIndex] = 1;
            }
        }
    });
}

__global__ void
setStaticFragmentData(uint staticFragmentNum, uint *canSwapFragmentD, uint *canSwapFragmentPrefixD,
                      uint *staticFragmentDataD) {
    streamVertices(staticFragmentNum, [&](uint index) {
        if (canSwapFragmentD[index] > 0) {
            staticFragmentDataD[canSwapFragmentPrefixD[index]] = index;
            canSwapFragmentD[index] = 0;
        }
    });
}

__global__ void
setFragmentDataOpt(uint *staticFragmentData, uint staticFragmentNum, uint *staticFragmentVisitRecordsD) {
    streamVertices(staticFragmentNum, [&](uint index) {
        uint fragmentId = index;
        if (staticFragmentVisitRecordsD[fragmentId] > 3600) {
            staticFragmentData[fragmentId] = 1;
            staticFragmentVisitRecordsD[fragmentId] = 0;
        } else {
            staticFragmentData[fragmentId] = 0;
        }
    });
}

__global__ void
setFragmentDataOpt4Pr(uint *staticFragmentData, uint fragmentNum, uint *fragmentVisitRecordsD,
                      bool *isActiveFragmentD, uint *fragmentNormalMap2StaticD, uint maxStaticFragment) {
    streamVertices(fragmentNum, [&](uint fragmentId) {
        /*if (fragmentId == 887550) {
            printf("fragmentId 887550 record %d \n", fragmentVisitRecordsD[fragmentId]);
        }*/
        if (fragmentVisitRecordsD[fragmentId] > 3200) {
            isActiveFragmentD[fragmentId] = false;
            //fragmentVisitRecordsD[fragmentId] = 0;
        } else {
            isActiveFragmentD[fragmentId] = true;
            fragmentVisitRecordsD[fragmentId] = 0;
        }
        if (!isActiveFragmentD[fragmentId]) {
            uint staticFragmentIndex = fragmentNormalMap2StaticD[fragmentId];
            if (staticFragmentIndex < maxStaticFragment) {
                staticFragmentData[staticFragmentIndex] = 1;
            }
        }
    });
}

uint reduceBool(uint *resultD, bool *isActiveD, uint vertexSize, dim3 grid, dim3 block) {
    //printf("reduceBool \n");
    uint activeNodesNum = 0;
    int blockSize = block.x;
    reduceByBool<<<grid, block, block.x * sizeof(uint)>>>(vertexSize, isActiveD, resultD);
    reduceResult<56><<<1, 64, block.x * sizeof(uint)>>>(resultD);
    cudaMemcpy(&activeNodesNum, resultD, sizeof(uint), cudaMemcpyDeviceToHost);
    return activeNodesNum;
}

__device__ void reduceStreamVertices(int vertices, bool *rawData, uint *result) {

    extern __shared__ uint sdata[];
    uint tid = threadIdx.x;
    sdata[tid] = 0;
    for (auto i : grid_stride_range(0, vertices)) {
        sdata[tid] += rawData[i];
    }
    __syncthreads();
    if (blockDim.x > 512 && tid < 512) { sdata[tid] += sdata[tid + 512]; }
    __syncthreads();
    if (blockDim.x > 256 && tid < 256) { sdata[tid] += sdata[tid + 256]; }
    __syncthreads();
    if (blockDim.x > 128 && tid < 128) { sdata[tid] += sdata[tid + 128]; }
    __syncthreads();
    if (blockDim.x > 64 && tid < 64) { sdata[tid] += sdata[tid + 64]; }
    __syncthreads();
    if (tid < 32) { sdata[tid] += sdata[tid + 32]; }
    __syncthreads();

    if (tid < 16) { sdata[tid] += sdata[tid + 16]; }
    __syncthreads();
    if (tid < 8) { sdata[tid] += sdata[tid + 8]; }
    __syncthreads();
    if (tid < 4) { sdata[tid] += sdata[tid + 4]; }
    __syncthreads();
    if (tid < 2) { sdata[tid] += sdata[tid + 2]; }
    if (tid < 1) { sdata[tid] += sdata[tid + 1]; }
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

__global__ void reduceByBool(uint vertexSize, bool *rawData, uint *result) {
    reduceStreamVertices(vertexSize, rawData, result);
}

template<int blockSize>
__global__ void reduceResult(uint *result) {
    extern __shared__ uint sdata[];
    uint tid = threadIdx.x;
    sdata[tid] = 0;
    if (tid < blockSize) {
        sdata[tid] = result[tid];
    }
    __syncthreads();
    if (tid < 32) { sdata[tid] += sdata[tid + 32]; }
    __syncthreads();
    if (tid < 16) { sdata[tid] += sdata[tid + 16]; }
    __syncthreads();
    if (tid < 8) { sdata[tid] += sdata[tid + 8]; }
    __syncthreads();
    if (tid < 4) { sdata[tid] += sdata[tid + 4]; }
    __syncthreads();
    if (tid < 2) { sdata[tid] += sdata[tid + 2]; }
    if (tid < 1) { sdata[tid] += sdata[tid + 1]; }
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

template<int BS> __global__ void scanWarpReduceInBlock(int n, bool* in, uint* out) {
    extern __shared__ uint sdata[];
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadIdx.x / 32;
    int idInWarp = threadIdx.x % 32;
    bool data = in[id];
    int sumInWarp = reduceInWarp<32>(idInWarp, data);
    if (idInWarp == 0) sdata[warpId] = sumInWarp;
    __syncthreads();
}

template<int NT>
__device__ int reduceInWarp(int idInWarp, bool data) {
    int ret = data;
    for (int i = NT / 2; i > 0; i /= 2) {
        data = __shfl_down(ret, i, NT);
        if (idInWarp < i) ret += data;
    }
    return ret;
}


