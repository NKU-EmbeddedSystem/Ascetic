## Ascetic
Ascetic is an out-of-GPU-memory graph processing framework.

#### Compilation
To compile the Ascetic. You need have cmake, g++ and CUDA 11.0 toolkit. 
You should enter the project root dir. 
Create a directory ie. cmake-build-debug. 
Enter it and cmake .. and make to complile the project.

#### Input graph formats
Ascetic accepts the binary CSR format just like :
```
0 4 7 9
1 2 3 4 2 3 4 3 4 5 6
```
There is an elegant tool which could transfer txt format to CSR format in the "converter" directory. But maybe you need to read it and fix it according to your need.

#### Running
```
$ ./ptgraph --input /dataset/somedata --type bfs --source 0
```

#### Publication
[ICPP'21] Ruiqi Tang, Ziyi Zhao, Kailun Wang, Xiaoli Gong, Jin Zhang, Wen-wen Wang, and Pen-Chung Yew. Ascetic: Enhancing Cross-Iterations Data Efficiency in Out-of-Memory Graph Processing on GPUs.




