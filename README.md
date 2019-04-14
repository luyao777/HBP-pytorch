# HBP-pytorch
![](<https://img.shields.io/badge/license-GPL--3.0-blue.svg>)![](<https://img.shields.io/badge/pytorch-%3E%3D0.4.0-yellow.svg>)![](<https://img.shields.io/badge/updated-april%2014%202019-green.svg>)

### **Overview**

A third-party reimplementation of Hierarchical Bilinear Pooling for Fine-Grained Visual Recognition in Pytorch.

The related paper is as follows:
    
    Hierarchical Bilinear Pooling for Fine-Grained Visual Recognition[C]
    Chaojian Yu, Xinyi Zhao, Qi Zheng, Peng Zhang, Xinge You*
    European Conference on Computer Vision. 2018.

Official Caffe implementation of Hierarchical Bilinear Pooling for Fine-Grained Visual Recognition is [HERE](https://github.com/ChaojianYu/Hierarchical-Bilinear-Pooling).

###  **Result**

---

| file       |  acc  |
| ---------- | :---: |
| HBP_fc     | 80.42 |
| HBP_fc_new | 79.79 |
| HBP_all    | 80.42 |

*Note that `HBP_fc_new.py` may be the closest to the original implementation. But it still doesn't work well.*

### **Last**

---

Based on my code and experimental results, it is far from the result of the original author. So you can use it as a reference for learning.

This code borrows from [HERE](https://github.com/HaoMood/bilinear-cnn). If you have any suggestions please contact me, I am still continue to improve the results.

Happy coding.
