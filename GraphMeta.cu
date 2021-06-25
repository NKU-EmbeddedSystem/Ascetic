//
// Created by gxl on 2021/2/1.
//

#include <fstream>
#include "GraphMeta.cuh"

/*template<class EdgeType>
void GraphMeta<EdgeType>::readDataFromFile(const string &fileName, bool isPagerank) {
    cout << "====== readDataFromFile ============" << endl;
    auto startTime = chrono::steady_clock::now();
    ifstream infile(fileName, ios::in | ios::binary);
    infile.read((char *) &this->vertexArrSize, sizeof(uint));
    infile.read((char *) &this->edgeArrSize, sizeof(uint));
    cout << "vertex num: " << this->vertexArrSize << " edge num: " << this->edgeArrSize << endl;
    outDegree = new uint[vertexArrSize];
    if (isPagerank) {
        infile.read((char *) outDegree, sizeof(uint) * vertexArrSize);
    }
    nodePointers = new uint[vertexArrSize];
    infile.read((char *) nodePointers, sizeof(uint) * vertexArrSize);
    edgeArray = new EdgeType[edgeArrSize];
    infile.read((char *) edgeArray, sizeof(EdgeType) * edgeArrSize);
    infile.close();
    auto endTime = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
    cout << "readDataFromFile " << duration << " ms" << endl;
}

*//*template<class EdgeType>
GraphMeta<EdgeType>::~GraphMeta() {
    cout << "~GraphMeta" << endl;
    delete[] nodePointers;
    delete[] outDegree;
    delete[] edgeArray;
}*//*

template<class EdgeType>
void GraphMeta<EdgeType>::initGraphHost() {
    degree = new SIZE_TYPE[vertexArrSize];
    isInStatic = new bool[vertexArrSize];
    overloadNodeList = new SIZE_TYPE[vertexArrSize];
    staticActiveNodeList = new SIZE_TYPE[vertexArrSize];
    activeOverloadNodePointers = new SIZE_TYPE[vertexArrSize];

    for (SIZE_TYPE i = 0; i < vertexArrSize - 1; i++) {
        if (nodePointers[i] > edgeArrSize) {
            cout << i << "   " << nodePointers[i] << endl;
            break;
        }
        degree[i] = nodePointers[i + 1] - nodePointers[i];
    }
    degree[vertexArrSize - 1] = edgeArrSize - nodePointers[vertexArrSize - 1];
    getMaxPartitionSize();
    initLableAndValue();
    overloadEdgeList = (SIZE_TYPE *) malloc(overloadSize * sizeof(SIZE_TYPE));
}


template<class EdgeType>
void GraphMeta<EdgeType>::initGraphDevice() {
    //pre store
    cudaMalloc(&staticEdgeListD, max_partition_size * sizeof(EdgeType));
    cudaMemcpy(staticEdgeListD, edgeArray, max_partition_size * sizeof(EdgeType), cudaMemcpyHostToDevice);

    cudaMalloc(&isInStaticD, vertexArrSize * sizeof(bool));
    cudaMalloc(&overloadNodeListD, vertexArrSize * sizeof(SIZE_TYPE));
    cudaMalloc(&staticNodePointerD, vertexArrSize * sizeof(SIZE_TYPE));
    cudaMemcpy(staticNodePointerD, nodePointers,vertexArrSize * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(isInStaticD, isInStatic, vertexArrSize * sizeof(bool), cudaMemcpyHostToDevice);

}

template<class EdgeType>
void GraphMeta<EdgeType>::initAndSetStaticNodePointers() {
    staticNodePointer = new uint[vertexArrSize];
    memcpy(staticNodePointer, nodePointers, vertexArrSize * sizeof(SIZE_TYPE));
    cudaMalloc(&staticNodePointerD, vertexArrSize * sizeof(SIZE_TYPE));
    cudaMemcpy(staticNodePointerD, nodePointers, vertexArrSize * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
    cudaMalloc(&overloadEdgeListD, partOverloadSize * sizeof(SIZE_TYPE));
    cudaMalloc(&degreeD, vertexArrSize * sizeof(SIZE_TYPE));
    cudaMalloc(&isActiveD, vertexArrSize * sizeof(bool));
    cudaMalloc(&isStaticActive, vertexArrSize * sizeof(bool));
    cudaMalloc(&isOverloadActive, vertexArrSize * sizeof(bool));
    cudaMalloc(&valueD, vertexArrSize * sizeof(SIZE_TYPE));
    cudaMalloc(&activeNodeLabelingPrefixD, vertexArrSize * sizeof(SIZE_TYPE));
    cudaMalloc(&overloadLabelingPrefixD, vertexArrSize * sizeof(SIZE_TYPE));

    cudaMalloc(&activeNodeListD, vertexArrSize * sizeof(SIZE_TYPE));
    cudaMalloc(&activeOverloadNodePointersD, vertexArrSize * sizeof(SIZE_TYPE));
    cudaMalloc(&activeOverloadDegreeD, vertexArrSize * sizeof(SIZE_TYPE));
    cudaMemcpy(degreeD, degree, vertexArrSize * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(valueD, value, vertexArrSize * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(isActiveD, label, vertexArrSize * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
    cudaMemset(isStaticActive, 0, vertexArrSize * sizeof(SIZE_TYPE));
    cudaMemset(isOverloadActive, 0, vertexArrSize * sizeof(SIZE_TYPE));
}


template<class EdgeType>
void GraphMeta<EdgeType>::getMaxPartitionSize() {
    int deviceID;
    cudaDeviceProp dev{};
    cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&dev, deviceID);
    size_t totalMemory;
    size_t availMemory;
    cudaMemGetInfo(&availMemory, &totalMemory);
    long reduceMem = paramSize * sizeof(uint) * (long) vertexArrSize;
    cout << "reduceMem " << reduceMem << " testNumNodes " << vertexArrSize << " ParamsSize " << paramSize
         << endl;
    total_gpu_size = (availMemory - reduceMem) / sizeof(EdgeType);

    //float adviseK = (10 - (float) edgeListSize / (float) totalSize) / 9;
    //uint dynamicDataMax = edgeListSize * edgeSize -
    float adviseK = (10 - (float) edgeArrSize / (float) total_gpu_size) / 9;
    cout << "adviseK " << adviseK << endl;
    if (adviseK < 0) {
        adviseK = 0.5;
        cout << "adviseK " << adviseK << endl;
    }
    if (adviseK > 1) {
        adviseK = 0.95;
        cout << "adviseK " << adviseK << endl;
    }
    float adviseRate = 0;
    if (adviseRate > 0) {
        adviseK = adviseRate;
    }

    max_partition_size = adviseK * total_gpu_size;
    cout << "availMemory " << availMemory << " totalMemory " << totalMemory << endl;
    printf("available memory is %ld totalGlobalMem is %ld, max static edge size is %ld\n total edge size %ld \n multiprocessors %d adviseK %f\n",
           availMemory - reduceMem,
           dev.totalGlobalMem, max_partition_size, total_gpu_size, dev.multiProcessorCount, adviseK);
    if (max_partition_size > UINT_MAX) {
        printf("bigger than DIST_INFINITY\n");
        max_partition_size = UINT_MAX;
    }
    uint temp = max_partition_size % fragmentSize;
    max_partition_size = max_partition_size - temp;
    max_static_node = 0;
    SIZE_TYPE edgesInStatic = 0;
    for (uint i = 0; i < vertexArrSize; i++) {
        if (nodePointers[i] < max_partition_size && (nodePointers[i] + degree[i] - 1) < max_partition_size) {
            isInStatic[i] = true;
            if (i > max_static_node) max_static_node = i;
            edgesInStatic += degree[i];
        } else {
            isInStatic[i] = false;
        }
    }

    partOverloadSize = total_gpu_size - max_partition_size;
    overloadSize = edgeArrSize - edgesInStatic;
}

template<class EdgeType>
void GraphMeta<EdgeType>::initLableAndValue() {

    label = new uint[vertexArrSize];
    if (algType == PR) {
        valuePr = new float[vertexArrSize];
        for (uint i = 0; i < vertexArrSize; i++) {
            label[i] = 1;
            valuePr[i] = 1.0;
        }
    } else {
        value = new SIZE_TYPE[vertexArrSize];
        switch (algType) {
            case BFS:
                for (uint i = 0; i < vertexArrSize; i++) {
                    label[i] = 0;
                    value[i] = UINT_MAX;
                }
                label[sourceNode] = 1;
                value[sourceNode] = 1;
                break;
            case SSSP:
                for (uint i = 0; i < vertexArrSize; i++) {
                    label[i] = 0;
                    value[i] = UINT_MAX;
                }
                label[sourceNode] = 1;
                value[sourceNode] = 1;
                break;
            case CC:
                for (uint i = 0; i < vertexArrSize; i++) {
                    label[i] = 1;
                    value[i] = i;
                }
        }
    }
}

*/
