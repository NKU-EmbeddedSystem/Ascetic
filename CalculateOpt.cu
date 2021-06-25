//
// Created by gxl on 2021/3/24.
//

#include "CalculateOpt.cuh"

template<class T>
__global__ void testRefresh(bool *isActive, T *data, uint size) {
    streamVertices(size, [&](uint id) {
        data[id] = isActive[id];
    });
}

void bfs_opt(string path, uint sourceNode, double adviseRate) {
    cout << "======bfs_opt=======" << endl;
    GraphMeta<uint> graph;
    graph.setAlgType(BFS);
    graph.setSourceNode(sourceNode);
    graph.readDataFromFile(path, false);
    graph.setPrestoreRatio(adviseRate, 13);
    graph.initGraphHost();
    graph.initGraphDevice();
    cudaDeviceSynchronize();
    cout << "1test!!!!!!!!!" <<endl;

    // by testing, reduceBool not better than thrust
    uint activeNodesNum = reduceBool(graph.resultD, graph.isActiveD, graph.vertexArrSize, graph.grid, graph.block);

    cout << "2test!!!!!!!!!" <<endl;
    TimeRecord<chrono::milliseconds> forULLProcess("forULLProcess");
    TimeRecord<chrono::milliseconds> preProcess("preProcess");
    TimeRecord<chrono::milliseconds> staticProcess("staticProcess");
    TimeRecord<chrono::milliseconds> overloadProcess("overloadProcess");
    TimeRecord<chrono::milliseconds> overloadMoveProcess("overloadMoveProcess");

    TimeRecord<chrono::milliseconds> totalProcess("totalProcess");
    totalProcess.startRecord();
    cout << "graph.vertexArrSize " << graph.vertexArrSize << endl;
    activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize, 0,
                                    thrust::plus<uint>());
    
    cout << "3test!!!!!!!!!" <<endl;
    //cout << "activeNodesNum " << activeNodesNum << endl;
    totalProcess.endRecord();
    totalProcess.print();
    totalProcess.clearRecord();

    EDGE_POINTER_TYPE overloadEdges = 0; 

    int testTimes = 1;
    for (int testIndex = 0; testIndex < testTimes; testIndex++) {
        graph.refreshLabelAndValue();
        cudaDeviceSynchronize();
        activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize, 0,
                                        thrust::plus<uint>());
        uint nodeSum = activeNodesNum;
        //cout << "activeNodesNum " << activeNodesNum << endl;
        int iter = 0;
        totalProcess.startRecord();
        while (activeNodesNum) {
            iter++;
            preProcess.startRecord();
            setStaticAndOverloadLabelBool<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.isActiveD,
                                                                       graph.isStaticActive, graph.isOverloadActive,
                                                                       graph.isInStaticD);
            uint staticNodeNum = thrust::reduce(graph.actStaticLablingThrust,
                                                graph.actStaticLablingThrust + graph.vertexArrSize, 0,
                                                thrust::plus<uint>());
            if (staticNodeNum > 0) {
                thrust::device_ptr<uint> tempTestPrefixThrust = thrust::device_ptr<uint>(graph.prefixSumTemp);
                    
                thrust::exclusive_scan(graph.actStaticLablingThrust, graph.actStaticLablingThrust + graph.vertexArrSize,
                                       tempTestPrefixThrust, 0, thrust::plus<uint>());
                setStaticActiveNodeArray<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.staticNodeListD,
                                                                      graph.isStaticActive,
                                                                      graph.prefixSumTemp);
                //cout << "iter " << iter << " staticNodeNum is " << staticNodeNum << endl;
            }

            uint overloadNodeNum = thrust::reduce(graph.actOverLablingThrust,
                                                  graph.actOverLablingThrust + graph.vertexArrSize, 0,
                                                  thrust::plus<uint>());
            EDGE_POINTER_TYPE overloadEdgeNum = 0;
            if (overloadNodeNum > 0) {
                thrust::device_ptr<uint> tempTestPrefixThrust = thrust::device_ptr<uint>(graph.prefixSumTemp);
                thrust::exclusive_scan(graph.actOverLablingThrust, graph.actOverLablingThrust + graph.vertexArrSize,
                                       tempTestPrefixThrust, 0, thrust::plus<uint>());
                setOverloadNodePointerSwap<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.overloadNodeListD,
                                                                        graph.activeOverloadDegreeD,
                                                                        graph.isOverloadActive,
                                                                        graph.prefixSumTemp, graph.degreeD);
                if(typeid(EDGE_POINTER_TYPE) == typeid(unsigned long long)) {
                    forULLProcess.startRecord();
                    cudaMemcpyAsync(graph.overloadNodeList, graph.overloadNodeListD, overloadNodeNum * sizeof(uint),
                                    cudaMemcpyDeviceToHost, graph.streamDynamic);
                    EDGE_POINTER_TYPE * overloadDegree = new EDGE_POINTER_TYPE[overloadNodeNum];
                    cudaMemcpyAsync(overloadDegree, graph.activeOverloadDegreeD, overloadNodeNum * sizeof(EDGE_POINTER_TYPE),
                                    cudaMemcpyDeviceToHost, graph.streamDynamic);
                    unsigned long long ankor = 0;
                    for(unsigned i = 0; i < overloadNodeNum; ++i) {
                        overloadEdgeNum += graph.degree[graph.overloadNodeList[i]];
                        if(i > 0) {
                            ankor += overloadDegree[i - 1];
                        }
                        graph.activeOverloadNodePointers[i] = ankor;
                        if(graph.activeOverloadNodePointers[i] > graph.edgeArrSize) {
                            cout << i << " : " << graph.activeOverloadNodePointers[i];
                        }
                    }
                    cudaMemcpyAsync(graph.activeOverloadNodePointersD, graph.activeOverloadNodePointers, overloadNodeNum * sizeof(EDGE_POINTER_TYPE),
                                    cudaMemcpyHostToDevice, graph.streamDynamic);
                    forULLProcess.endRecord();
                } else {
                    thrust::device_ptr<EDGE_POINTER_TYPE> tempTestNodePointersThrust = thrust::device_ptr<EDGE_POINTER_TYPE>(graph.activeOverloadNodePointersD);
                    thrust::exclusive_scan(graph.actOverDegreeThrust, graph.actOverDegreeThrust + graph.vertexArrSize,
                                           tempTestNodePointersThrust, 0, thrust::plus<uint>());
                    overloadEdgeNum = thrust::reduce(thrust::device, graph.activeOverloadDegreeD,
                                                     graph.activeOverloadDegreeD + overloadNodeNum, 0);
                }
                
                overloadEdges += overloadEdgeNum;
               // cout << "iter " << iter << " overloadNodeNum is " << overloadNodeNum << endl;
                //cout << "iter " << iter << " overloadEdgeNum is " << overloadEdgeNum << endl;
            }

            if (staticNodeNum > 0) {
                setLabelDefaultOpt<<<graph.grid, graph.block, 0, graph.steamStatic>>>(staticNodeNum,
                                                                                      graph.staticNodeListD,
                                                                                      graph.isActiveD);
            }
            if (overloadNodeNum > 0) {
                setLabelDefaultOpt<<<graph.grid, graph.block, 0, graph.steamStatic>>>(overloadNodeNum,
                                                                                      graph.overloadNodeListD,
                                                                                      graph.isActiveD);
            }

            preProcess.endRecord();
            staticProcess.startRecord();
            bfs_kernelStatic<<<graph.grid, graph.block, 0, graph.steamStatic>>>(staticNodeNum, graph.staticNodeListD,
                                                                                graph.staticNodePointerD, graph.degreeD,
                                                                                graph.staticEdgeListD, graph.valueD,
                                                                                graph.isActiveD);
            //cudaDeviceSynchronize();
            gpuErrorcheck(cudaPeekAtLastError());

            if (overloadNodeNum > 0) {
                cudaMemcpyAsync(graph.overloadNodeList, graph.overloadNodeListD, overloadNodeNum * sizeof(uint),
                                cudaMemcpyDeviceToHost, graph.streamDynamic);
                cudaMemcpyAsync(graph.activeOverloadNodePointers, graph.activeOverloadNodePointersD,
                                overloadNodeNum * sizeof(EDGE_POINTER_TYPE),
                                cudaMemcpyDeviceToHost, graph.streamDynamic);
                graph.fillEdgeArrByMultiThread(overloadNodeNum);
                graph.caculatePartInfoForEdgeList(overloadNodeNum, overloadEdgeNum);

                cudaDeviceSynchronize();

                gpuErrorcheck(cudaPeekAtLastError());
                staticProcess.endRecord();

                overloadProcess.startRecord();
                for (int i = 0; i < graph.partEdgeListInfoArr.size(); i++) {
                    // << i << " graph.activeOverloadNodePointers[graph.partEdgeListInfoArr[i].partStartIndex] " << graph.activeOverloadNodePointers[graph.partEdgeListInfoArr[i].partStartIndex] << endl;
                    //cout << i << " graph.partEdgeListInfoArr[i].partEdgeNums " << graph.partEdgeListInfoArr[i].partEdgeNums << endl;
                    overloadMoveProcess.startRecord();
                    gpuErrorcheck(cudaMemcpy(graph.overloadEdgeListD, graph.overloadEdgeList +
                                                                      graph.activeOverloadNodePointers[graph.partEdgeListInfoArr[i].partStartIndex],
                                             graph.partEdgeListInfoArr[i].partEdgeNums * sizeof(uint),
                                             cudaMemcpyHostToDevice))
                    overloadMoveProcess.endRecord();
                    bfs_kernelDynamicPart<<<graph.grid, graph.block, 0, graph.streamDynamic>>>(
                            graph.partEdgeListInfoArr[i].partStartIndex,
                            graph.partEdgeListInfoArr[i].partActiveNodeNums,
                            graph.overloadNodeListD, graph.degreeD,
                            graph.valueD, graph.isActiveD,
                            graph.overloadEdgeListD,
                            graph.activeOverloadNodePointersD);
                    cudaDeviceSynchronize();
                }
                overloadProcess.endRecord();

            } else {
                cudaDeviceSynchronize();
                staticProcess.endRecord();
            }
            cudaDeviceSynchronize();
            preProcess.startRecord();
            activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize,
                                            0,
                                            thrust::plus<uint>());
            nodeSum += activeNodesNum;
            preProcess.endRecord();
            cout << "iter " << iter << " activeNodesNum " << activeNodesNum << endl;
            //break;

        }
        totalProcess.endRecord();
        totalProcess.print();
        preProcess.print();
        staticProcess.print();
        overloadProcess.print();
        forULLProcess.print();
        overloadMoveProcess.print();
        cout << "nodeSum : " << nodeSum << endl;
        cout << "move overload size : " << overloadEdges * sizeof(uint) << endl;
    }
    gpuErrorcheck(cudaPeekAtLastError());
}

void cc_opt(string path, double adviseRate) {
    cout << "==========cc_opt==========" << endl;
    GraphMeta<uint> graph;
    graph.setAlgType(CC);
    graph.readDataFromFile(path, false);
    graph.setPrestoreRatio(adviseRate, 13);
    graph.initGraphHost();
    graph.initGraphDevice();
    cudaDeviceSynchronize();


    // by testing, reduceBool not better than thrust
    uint activeNodesNum = reduceBool(graph.resultD, graph.isActiveD, graph.vertexArrSize, graph.grid, graph.block);
    TimeRecord<chrono::milliseconds> totalProcess("totalProcess");
    TimeRecord<chrono::milliseconds> forULLProcess("forULLProcess");
    TimeRecord<chrono::milliseconds> fillProcess("fillProcess");
    TimeRecord<chrono::milliseconds> staticProcess("staticProcess");
    TimeRecord<chrono::milliseconds> overloadProcess("overloadProcess");

    TimeRecord<chrono::milliseconds> overloadMoveProcess("overloadMoveProcess");

    totalProcess.startRecord();
    activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize, 0,
                                    thrust::plus<uint>());
    totalProcess.endRecord();
    cout << "activeNodesNum " << activeNodesNum << endl;
    totalProcess.print();
    totalProcess.clearRecord();
    int testTimes = 1;
    EDGE_POINTER_TYPE overloadEdges = 0;
    for (int testIndex = 0; testIndex < testTimes; testIndex++) {
        graph.refreshLabelAndValue();
        cudaDeviceSynchronize();
        activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize, 0,
                                        thrust::plus<uint>());
        uint nodeSum = activeNodesNum;
        int iter = 0;

        totalProcess.startRecord();
        while (activeNodesNum) {
            iter++;
            setStaticAndOverloadLabelBool<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.isActiveD,
                                                                       graph.isStaticActive, graph.isOverloadActive,
                                                                       graph.isInStaticD);
            uint staticNodeNum = thrust::reduce(graph.actStaticLablingThrust,
                                                graph.actStaticLablingThrust + graph.vertexArrSize, 0,
                                                thrust::plus<uint>());
            if (staticNodeNum > 0) {
                thrust::device_ptr<uint> tempTestPrefixThrust = thrust::device_ptr<uint>(graph.prefixSumTemp);
                thrust::exclusive_scan(graph.actStaticLablingThrust, graph.actStaticLablingThrust + graph.vertexArrSize,
                                       tempTestPrefixThrust, 0, thrust::plus<uint>());
                setStaticActiveNodeArray<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.staticNodeListD,
                                                                      graph.isStaticActive,
                                                                      graph.prefixSumTemp);
                //cout << "iter " << iter << " staticNodeNum is " << staticNodeNum << endl;
            }

            uint overloadNodeNum = thrust::reduce(graph.actOverLablingThrust,
                                                  graph.actOverLablingThrust + graph.vertexArrSize, 0,
                                                  thrust::plus<uint>());
            EDGE_POINTER_TYPE overloadEdgeNum = 0;
            if (overloadNodeNum > 0) {
                thrust::device_ptr<uint> tempTestPrefixThrust = thrust::device_ptr<uint>(graph.prefixSumTemp);
                thrust::exclusive_scan(graph.actOverLablingThrust, graph.actOverLablingThrust + graph.vertexArrSize,
                                       tempTestPrefixThrust, 0, thrust::plus<uint>());
                setOverloadNodePointerSwap<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.overloadNodeListD,
                                                                        graph.activeOverloadDegreeD,
                                                                        graph.isOverloadActive,
                                                                        graph.prefixSumTemp, graph.degreeD);
                
                if(typeid(EDGE_POINTER_TYPE) == typeid(unsigned long long)) {
                    forULLProcess.startRecord();
                    cudaMemcpyAsync(graph.overloadNodeList, graph.overloadNodeListD, overloadNodeNum * sizeof(uint),
                                    cudaMemcpyDeviceToHost, graph.streamDynamic);
                    EDGE_POINTER_TYPE * overloadDegree = new EDGE_POINTER_TYPE[overloadNodeNum];
                    cudaMemcpyAsync(overloadDegree, graph.activeOverloadDegreeD, overloadNodeNum * sizeof(EDGE_POINTER_TYPE),
                                    cudaMemcpyDeviceToHost, graph.streamDynamic);
                    unsigned long long ankor = 0;
                    for(unsigned i = 0; i < overloadNodeNum; ++i) {
                        overloadEdgeNum += graph.degree[graph.overloadNodeList[i]];
                        if(i > 0) {
                            ankor += overloadDegree[i - 1];
                        }
                        graph.activeOverloadNodePointers[i] = ankor;
                        if(graph.activeOverloadNodePointers[i] > graph.edgeArrSize) {
                            cout << i << " : " << graph.activeOverloadNodePointers[i];
                        }
                    }
                    cudaMemcpyAsync(graph.activeOverloadNodePointersD, graph.activeOverloadNodePointers, overloadNodeNum * sizeof(EDGE_POINTER_TYPE),
                                    cudaMemcpyHostToDevice, graph.streamDynamic);
                    forULLProcess.endRecord();
                } else {
                    thrust::device_ptr<EDGE_POINTER_TYPE> tempTestNodePointersThrust = thrust::device_ptr<EDGE_POINTER_TYPE>(graph.activeOverloadNodePointersD);
                    thrust::exclusive_scan(graph.actOverDegreeThrust, graph.actOverDegreeThrust + graph.vertexArrSize,
                                           tempTestNodePointersThrust, 0, thrust::plus<uint>());
                    overloadEdgeNum = thrust::reduce(thrust::device, graph.activeOverloadDegreeD,
                                                     graph.activeOverloadDegreeD + overloadNodeNum, 0);
                }

                overloadEdges += overloadEdgeNum;
                //cout << "iter " << iter << " overloadNodeNum is " << overloadNodeNum << endl;
                //cout << "iter " << iter << " overloadEdgeNum is " << overloadEdgeNum << endl;
            }

            if (staticNodeNum > 0) {
                setLabelDefaultOpt<<<graph.grid, graph.block, 0, graph.steamStatic>>>(staticNodeNum,
                                                                                      graph.staticNodeListD,
                                                                                      graph.isActiveD);
            }
            if (overloadNodeNum > 0) {
                setLabelDefaultOpt<<<graph.grid, graph.block, 0, graph.steamStatic>>>(overloadNodeNum,
                                                                                      graph.overloadNodeListD,
                                                                                      graph.isActiveD);
            }

            fillProcess.startRecord();
            cc_kernelStaticSwap<<<graph.grid, graph.block, 0, graph.steamStatic>>>(staticNodeNum, graph.staticNodeListD,
                                                                                   graph.staticNodePointerD,
                                                                                   graph.degreeD,
                                                                                   graph.staticEdgeListD, graph.valueD,
                                                                                   graph.isActiveD, graph.isInStaticD);
            //cudaDeviceSynchronize();

            if (overloadNodeNum > 0) {
                cudaMemcpyAsync(graph.overloadNodeList, graph.overloadNodeListD, overloadNodeNum * sizeof(uint),
                                cudaMemcpyDeviceToHost, graph.streamDynamic);
                cudaMemcpyAsync(graph.activeOverloadNodePointers, graph.activeOverloadNodePointersD,
                                overloadNodeNum * sizeof(EDGE_POINTER_TYPE),
                                cudaMemcpyDeviceToHost, graph.streamDynamic);
                graph.fillEdgeArrByMultiThread(overloadNodeNum);
                graph.caculatePartInfoForEdgeList(overloadNodeNum, overloadEdgeNum);

                cudaDeviceSynchronize();
                fillProcess.endRecord();

                for (int i = 0; i < graph.partEdgeListInfoArr.size(); i++) {
                    overloadMoveProcess.startRecord();
                    gpuErrorcheck(cudaMemcpy(graph.overloadEdgeListD, graph.overloadEdgeList +
                                                                      graph.activeOverloadNodePointers[graph.partEdgeListInfoArr[i].partStartIndex],
                                             graph.partEdgeListInfoArr[i].partEdgeNums * sizeof(uint),
                                             cudaMemcpyHostToDevice))
                    overloadMoveProcess.endRecord();
                    cc_kernelDynamicSwap<<<graph.grid, graph.block, 0, graph.streamDynamic>>>(
                            graph.partEdgeListInfoArr[i].partStartIndex,
                            graph.partEdgeListInfoArr[i].partActiveNodeNums,
                            graph.overloadNodeListD, graph.degreeD,
                            graph.valueD, graph.isActiveD,
                            graph.overloadEdgeListD,
                            graph.activeOverloadNodePointersD);
                    cudaDeviceSynchronize();
                }

            } else {
                cudaDeviceSynchronize();
            }
            cudaDeviceSynchronize();
            activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize,
                                            0,
                                            thrust::plus<uint>());
            nodeSum += activeNodesNum;
            //cout << "iter " << iter << " activeNodesNum " << activeNodesNum << endl;

        }
        totalProcess.endRecord();
        totalProcess.print();
        forULLProcess.print();
        fillProcess.print();
        overloadMoveProcess.print();
        cout << "nodeSum : " << nodeSum << endl;
        cout << "move overload size : " << overloadEdges * sizeof(uint) << endl;
    }
    gpuErrorcheck(cudaPeekAtLastError());
}

void sssp_opt(string path, uint sourceNode, double adviseRate) {
    cout << "========sssp_opt==========" << endl;
    GraphMeta<EdgeWithWeight> graph;
    graph.setAlgType(SSSP);
    graph.setSourceNode(sourceNode);
    graph.readDataFromFile(path, false);
    graph.setPrestoreRatio(adviseRate, 15);
    graph.initGraphHost();
    graph.initGraphDevice();
    cudaDeviceSynchronize();


    // by testing, reduceBool not better than thrust
    uint activeNodesNum = reduceBool(graph.resultD, graph.isActiveD, graph.vertexArrSize, graph.grid, graph.block);
    TimeRecord<chrono::milliseconds> totalProcess("totalProcess");
    TimeRecord<chrono::milliseconds> forULLProcess("forULLProcess");
    TimeRecord<chrono::milliseconds> overloadMoveProcess("overloadMoveProcess");
    totalProcess.startRecord();
    activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize, 0,
                                    thrust::plus<uint>());
    totalProcess.endRecord();
    cout << "activeNodesNum " << activeNodesNum << endl;
    totalProcess.print();
    totalProcess.clearRecord();
    int testTimes = 1;
    EDGE_POINTER_TYPE overloadEdges = 0;
    for (int testIndex = 0; testIndex < testTimes; testIndex++) {
        graph.refreshLabelAndValue();
        cudaDeviceSynchronize();
        activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize, 0,
                                        thrust::plus<uint>());
        uint nodeSum = activeNodesNum;
        int iter = 0;
        totalProcess.startRecord();
        while (activeNodesNum) {
            iter++;
            setStaticAndOverloadLabelBool<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.isActiveD,
                                                                       graph.isStaticActive, graph.isOverloadActive,
                                                                       graph.isInStaticD);
            uint staticNodeNum = thrust::reduce(graph.actStaticLablingThrust,
                                                graph.actStaticLablingThrust + graph.vertexArrSize, 0,
                                                thrust::plus<uint>());
            if (staticNodeNum > 0) {
                thrust::device_ptr<uint> tempTestPrefixThrust = thrust::device_ptr<uint>(graph.prefixSumTemp);
                thrust::exclusive_scan(graph.actStaticLablingThrust, graph.actStaticLablingThrust + graph.vertexArrSize,
                                       tempTestPrefixThrust, 0, thrust::plus<uint>());
                setStaticActiveNodeArray<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.staticNodeListD,
                                                                      graph.isStaticActive,
                                                                      graph.prefixSumTemp);
                //cout << "iter " << iter << " staticNodeNum is " << staticNodeNum << endl;
            }

            uint overloadNodeNum = thrust::reduce(graph.actOverLablingThrust,
                                                  graph.actOverLablingThrust + graph.vertexArrSize, 0,
                                                  thrust::plus<uint>());
            EDGE_POINTER_TYPE overloadEdgeNum = 0;
            if (overloadNodeNum > 0) {
                thrust::device_ptr<uint> tempTestPrefixThrust = thrust::device_ptr<uint>(graph.prefixSumTemp);
                thrust::exclusive_scan(graph.actOverLablingThrust, graph.actOverLablingThrust + graph.vertexArrSize,
                                       tempTestPrefixThrust, 0, thrust::plus<uint>());
                setOverloadNodePointerSwap<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.overloadNodeListD,
                                                                        graph.activeOverloadDegreeD,
                                                                        graph.isOverloadActive,
                                                                        graph.prefixSumTemp, graph.degreeD);
                if(typeid(EDGE_POINTER_TYPE) == typeid(unsigned long long)) {
                    forULLProcess.startRecord();
                    cudaMemcpyAsync(graph.overloadNodeList, graph.overloadNodeListD, overloadNodeNum * sizeof(uint),
                                    cudaMemcpyDeviceToHost, graph.streamDynamic);
                    EDGE_POINTER_TYPE * overloadDegree = new EDGE_POINTER_TYPE[overloadNodeNum];
                    cudaMemcpyAsync(overloadDegree, graph.activeOverloadDegreeD, overloadNodeNum * sizeof(EDGE_POINTER_TYPE),
                                    cudaMemcpyDeviceToHost, graph.streamDynamic);
                    unsigned long long ankor = 0;
                    for(unsigned i = 0; i < overloadNodeNum; ++i) {
                        overloadEdgeNum += graph.degree[graph.overloadNodeList[i]];
                        if(i > 0) {
                            ankor += overloadDegree[i - 1];
                        }
                        graph.activeOverloadNodePointers[i] = ankor;
                        if(graph.activeOverloadNodePointers[i] > graph.edgeArrSize) {
                            cout << i << " : " << graph.activeOverloadNodePointers[i];
                        }
                    }
                    cudaMemcpyAsync(graph.activeOverloadNodePointersD, graph.activeOverloadNodePointers, overloadNodeNum * sizeof(EDGE_POINTER_TYPE),
                                    cudaMemcpyHostToDevice, graph.streamDynamic);
                    forULLProcess.endRecord();
                } else {
                    thrust::device_ptr<EDGE_POINTER_TYPE> tempTestNodePointersThrust = thrust::device_ptr<EDGE_POINTER_TYPE>(graph.activeOverloadNodePointersD);

                    thrust::exclusive_scan(graph.actOverDegreeThrust, graph.actOverDegreeThrust + graph.vertexArrSize,
                                           tempTestNodePointersThrust, 0, thrust::plus<uint>());
                    overloadEdgeNum = thrust::reduce(thrust::device, graph.activeOverloadDegreeD,
                                                     graph.activeOverloadDegreeD + overloadNodeNum, 0);
                }
                overloadEdges += overloadEdgeNum;
                //cout << "iter " << iter << " overloadNodeNum is " << overloadNodeNum << endl;
                //cout << "iter " << iter << " overloadEdgeNum is " << overloadEdgeNum << endl;
            }

            if (staticNodeNum > 0) {
                setLabelDefaultOpt<<<graph.grid, graph.block, 0, graph.steamStatic>>>(staticNodeNum,
                                                                                      graph.staticNodeListD,
                                                                                      graph.isActiveD);
            }
            if (overloadNodeNum > 0) {
                setLabelDefaultOpt<<<graph.grid, graph.block, 0, graph.steamStatic>>>(overloadNodeNum,
                                                                                      graph.overloadNodeListD,
                                                                                      graph.isActiveD);
            }

            sssp_kernel<<<graph.grid, graph.block, 0, graph.steamStatic>>>(staticNodeNum, graph.staticNodeListD,
                                                                           graph.staticNodePointerD, graph.degreeD,
                                                                           graph.staticEdgeListD, graph.valueD,
                                                                           graph.isActiveD);
            //cudaDeviceSynchronize();

            if (overloadNodeNum > 0) {
                cudaMemcpyAsync(graph.overloadNodeList, graph.overloadNodeListD, overloadNodeNum * sizeof(uint),
                                cudaMemcpyDeviceToHost, graph.streamDynamic);
                cudaMemcpyAsync(graph.activeOverloadNodePointers, graph.activeOverloadNodePointersD,
                                overloadNodeNum * sizeof(EDGE_POINTER_TYPE),
                                cudaMemcpyDeviceToHost, graph.streamDynamic);
                graph.fillEdgeArrByMultiThread(overloadNodeNum);
                graph.caculatePartInfoForEdgeList(overloadNodeNum, overloadEdgeNum);
                cudaDeviceSynchronize();

                for (int i = 0; i < graph.partEdgeListInfoArr.size(); i++) {
                    //cout << "graph.partEdgeListInfoArr[i].partEdgeNums " << graph.partEdgeListInfoArr[i].partEdgeNums << endl;
                    overloadMoveProcess.startRecord();
                    gpuErrorcheck(cudaMemcpy(graph.overloadEdgeListD, graph.overloadEdgeList +
                                                                      graph.activeOverloadNodePointers[graph.partEdgeListInfoArr[i].partStartIndex],
                                             graph.partEdgeListInfoArr[i].partEdgeNums * sizeof(EdgeWithWeight),
                                             cudaMemcpyHostToDevice))
                    overloadMoveProcess.endRecord();
                    sssp_kernelDynamic<<<graph.grid, graph.block, 0, graph.streamDynamic>>>(
                            graph.partEdgeListInfoArr[i].partStartIndex,
                            graph.partEdgeListInfoArr[i].partActiveNodeNums,
                            graph.overloadNodeListD, graph.degreeD,
                            graph.valueD, graph.isActiveD,
                            graph.overloadEdgeListD,
                            graph.activeOverloadNodePointersD);
                    cudaDeviceSynchronize();
                }

            } else {
                cudaDeviceSynchronize();
            }
            cudaDeviceSynchronize();
            activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize,
                                            0,
                                            thrust::plus<uint>());
            nodeSum += activeNodesNum;
            cout << "iter " << iter << " activeNodesNum " << activeNodesNum << endl;

        }
        totalProcess.endRecord();
        totalProcess.print();
        forULLProcess.print();
        overloadMoveProcess.print();
        cout << "nodeSum : " << nodeSum << endl;
        cout << "move overload size : " << overloadEdges * sizeof(EdgeWithWeight) << endl;
    }
    gpuErrorcheck(cudaPeekAtLastError());
}

void pr_opt(string path, double adviseRate) {
    cout << "=======pr_opt=======" << endl;
    GraphMeta<uint> graph;
    graph.setAlgType(PR);
    graph.readDataFromFile(path, true);
    graph.setPrestoreRatio(adviseRate, 17);
    graph.initGraphHost();
    graph.initGraphDevice();
    cudaDeviceSynchronize();


    // by testing, reduceBool not better than thrust
    uint activeNodesNum = reduceBool(graph.resultD, graph.isActiveD, graph.vertexArrSize, graph.grid, graph.block);
    TimeRecord<chrono::milliseconds> preProcess("preProcess");
    TimeRecord<chrono::milliseconds> staticProcess("staticProcess");
    TimeRecord<chrono::milliseconds> overloadProcess("overloadProcess");
    TimeRecord<chrono::milliseconds> totalProcess("totalProcess");
    TimeRecord<chrono::milliseconds> forULLProcess("forULLProcess");
    TimeRecord<chrono::milliseconds> overloadMoveProcess("overloadMoveProcess");

    totalProcess.startRecord();

    graph.refreshLabelAndValue();
    
    activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize, 0,
                                    thrust::plus<uint>());
    totalProcess.endRecord();
    cout << "activeNodesNum " << activeNodesNum << endl;
    totalProcess.print();
    totalProcess.clearRecord();
    int testTimes = 1;
    EDGE_POINTER_TYPE overloadEdges = 0;
    for (int testIndex = 0; testIndex < testTimes; testIndex++) {
        activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize, 0,
                                thrust::plus<uint>());
        gpuErrorcheck(cudaPeekAtLastError());
        uint nodeSum = activeNodesNum;
        int iter = 0;
        double totalSum = thrust::reduce(thrust::device, graph.valuePrD, graph.valuePrD + graph.vertexArrSize) / graph.vertexArrSize;
        cout << "totalSum " << totalSum << endl;
        totalProcess.startRecord();
        gpuErrorcheck(cudaPeekAtLastError());
        cout << "iter " << iter << " activeNodesNum " << activeNodesNum << endl;
        while (activeNodesNum > 3) {
            break;
            iter++;
            preProcess.startRecord();
            setStaticAndOverloadLabelBool<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.isActiveD,
                                                                       graph.isStaticActive, graph.isOverloadActive,
                                                                       graph.isInStaticD);
            uint staticNodeNum = thrust::reduce(graph.actStaticLablingThrust,
                                                graph.actStaticLablingThrust + graph.vertexArrSize, 0,
                                                thrust::plus<uint>());
            if (staticNodeNum > 0) {
                thrust::device_ptr<uint> tempTestPrefixThrust = thrust::device_ptr<uint>(graph.prefixSumTemp);
                thrust::exclusive_scan(graph.actStaticLablingThrust, graph.actStaticLablingThrust + graph.vertexArrSize,
                                       tempTestPrefixThrust, 0, thrust::plus<uint>());
                setStaticActiveNodeArray<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.staticNodeListD,
                                                                      graph.isStaticActive,
                                                                      graph.prefixSumTemp);
                //cout << "iter " << iter << " staticNodeNum is " << staticNodeNum << endl;
            }

            uint overloadNodeNum = thrust::reduce(graph.actOverLablingThrust,
                                                  graph.actOverLablingThrust + graph.vertexArrSize, 0,
                                                  thrust::plus<uint>());
            EDGE_POINTER_TYPE overloadEdgeNum = 0;
            if (overloadNodeNum > 0) {
                thrust::device_ptr<uint> tempTestPrefixThrust = thrust::device_ptr<uint>(graph.prefixSumTemp);
                thrust::exclusive_scan(graph.actOverLablingThrust, graph.actOverLablingThrust + graph.vertexArrSize,
                                       tempTestPrefixThrust, 0, thrust::plus<uint>());
                setOverloadNodePointerSwap<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.overloadNodeListD,
                                                                        graph.activeOverloadDegreeD,
                                                                        graph.isOverloadActive,
                                                                        graph.prefixSumTemp, graph.degreeD);
                if(typeid(EDGE_POINTER_TYPE) == typeid(unsigned long long)) {
                    forULLProcess.startRecord();
                    cudaMemcpyAsync(graph.overloadNodeList, graph.overloadNodeListD, overloadNodeNum * sizeof(uint),
                                    cudaMemcpyDeviceToHost, graph.streamDynamic);
                    EDGE_POINTER_TYPE * overloadDegree = new EDGE_POINTER_TYPE[overloadNodeNum];
                    cudaMemcpyAsync(overloadDegree, graph.activeOverloadDegreeD, overloadNodeNum * sizeof(EDGE_POINTER_TYPE),
                                    cudaMemcpyDeviceToHost, graph.streamDynamic);
                    unsigned long long ankor = 0;
                    for(unsigned i = 0; i < overloadNodeNum; ++i) {
                        overloadEdgeNum += graph.degree[graph.overloadNodeList[i]];
                        if(i > 0) {
                            ankor += overloadDegree[i - 1];
                        }
                        graph.activeOverloadNodePointers[i] = ankor;
                        if(graph.activeOverloadNodePointers[i] > graph.edgeArrSize) {
                            cout << i << " : " << graph.activeOverloadNodePointers[i];
                        }
                    }
                    cudaMemcpyAsync(graph.activeOverloadNodePointersD, graph.activeOverloadNodePointers, overloadNodeNum * sizeof(EDGE_POINTER_TYPE),
                                    cudaMemcpyHostToDevice, graph.streamDynamic);
                    forULLProcess.endRecord();
                } else {
                    thrust::device_ptr<EDGE_POINTER_TYPE> tempTestNodePointersThrust = thrust::device_ptr<EDGE_POINTER_TYPE>(graph.activeOverloadNodePointersD);
                    
                    thrust::exclusive_scan(graph.actOverDegreeThrust, graph.actOverDegreeThrust + graph.vertexArrSize,
                                           tempTestNodePointersThrust, 0, thrust::plus<uint>());
                    overloadEdgeNum = thrust::reduce(thrust::device, graph.activeOverloadDegreeD,
                                                     graph.activeOverloadDegreeD + overloadNodeNum, 0);
                }
                overloadEdges += overloadEdgeNum;
                //cout << "iter " << iter << " overloadNodeNum is " << overloadNodeNum << endl;
                //cout << "iter " << iter << " overloadEdgeNum is " << overloadEdgeNum << endl;
            }

            preProcess.endRecord();
            staticProcess.startRecord();
            prSumKernel_static<<<graph.grid, graph.block, 0, graph.steamStatic>>>(staticNodeNum, graph.staticNodeListD,
                                                                                  graph.staticNodePointerD,
                                                                                  graph.staticEdgeListD, graph.degreeD,
                                                                                  graph.outDegreeD, graph.valuePrD,
                                                                                  graph.sumD);
            //cudaDeviceSynchronize();

            if (overloadNodeNum > 0) {
                cudaMemcpyAsync(graph.overloadNodeList, graph.overloadNodeListD, overloadNodeNum * sizeof(uint),
                                cudaMemcpyDeviceToHost, graph.streamDynamic);
                cudaMemcpyAsync(graph.activeOverloadNodePointers, graph.activeOverloadNodePointersD,
                                overloadNodeNum * sizeof(EDGE_POINTER_TYPE),
                                cudaMemcpyDeviceToHost, graph.streamDynamic);
                graph.fillEdgeArrByMultiThread(overloadNodeNum);
                graph.caculatePartInfoForEdgeList(overloadNodeNum, overloadEdgeNum);
                cudaDeviceSynchronize();
                staticProcess.endRecord();
                overloadProcess.startRecord();
                for (int i = 0; i < graph.partEdgeListInfoArr.size(); i++) {
                    overloadMoveProcess.startRecord();
                    gpuErrorcheck(cudaMemcpy(graph.overloadEdgeListD, graph.overloadEdgeList +
                                                                      graph.activeOverloadNodePointers[graph.partEdgeListInfoArr[i].partStartIndex],
                                             graph.partEdgeListInfoArr[i].partEdgeNums * sizeof(uint),
                                             cudaMemcpyHostToDevice))
                    overloadMoveProcess.endRecord();
                    prSumKernel_dynamic<<<graph.grid, graph.block, 0, graph.streamDynamic>>>(
                            graph.partEdgeListInfoArr[i].partStartIndex,
                            graph.partEdgeListInfoArr[i].partActiveNodeNums,
                            graph.overloadNodeListD,
                            graph.activeOverloadNodePointersD,
                            graph.overloadEdgeListD, graph.degreeD, graph.outDegreeD,
                            graph.valuePrD, graph.sumD);
                    cudaDeviceSynchronize();
                }
                overloadProcess.endRecord();

            } else {
                staticProcess.endRecord();
                cudaDeviceSynchronize();
            }

            preProcess.startRecord();
            prKernel_Opt<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.valuePrD, graph.sumD, graph.isActiveD);
            cudaDeviceSynchronize();
            activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize,
                                            0,
                                            thrust::plus<uint>());
            nodeSum += activeNodesNum;
            preProcess.endRecord();
            cout << "iter " << iter << " activeNodesNum " << activeNodesNum << endl;

        }

        totalProcess.endRecord();
        totalProcess.print();
        preProcess.print();
        staticProcess.print();
        overloadProcess.print();
        forULLProcess.print();
        overloadMoveProcess.print();
        cout << "nodeSum : " << nodeSum << endl;
        cout << "move overload size : " << overloadEdges * sizeof(uint) << endl;
    }
    gpuErrorcheck(cudaPeekAtLastError());
}