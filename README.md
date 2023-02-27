# PaRK-Detect: Towards Efficient Multi-Task Satellite Imagery Road Extraction via Patch-Wise Keypoints Detection
Automatically extracting roads from satellite imagery is a fundamental yet challenging computer vision task in the field of remote sensing. 
Pixel-wise semantic segmentation-based approaches and graph-based approaches are two prevailing schemes. However, prior works show the 
imperfections that semantic segmentation-based approaches yield road graphs with low connectivity, while graph-based methods with iterative 
exploring paradigms and smaller receptive fields focus more on local information and are also time-consuming. In this paper, we propose a 
new scheme for multi-task satellite imagery road extraction, Patch-wise Road Keypoints Detection (PaRK-Detect). Building on top of D-LinkNet 
architecture and adopting the structure of keypoint detection, our framework predicts the position of patch-wise road keypoints and the 
adjacent relationships between them to construct road graphs in a single pass. Meanwhile, the multi-task framework also performs pixel-wise 
semantic segmentation and generates road segmentation masks. We evaluate our approach against the existing state-of-the-art methods on 
DeepGlobe, Massachusetts Roads, and RoadTracer datasets and achieve competitive or better results. We also demonstrate a considerable 
outperformance in terms of inference speed.

```
@title = {PaRK-Detect: Towards Efficient Multi-Task Satellite Imagery Road Extraction via Patch-Wise Keypoints Detection},  
@author = {Shenwei Xie (BUPT PRIS)}
@time = {from 2021/11/01}
@publication = {BMVC 2022 (oral)}
```

## 0. Introduction and Related Papers

## Framework
![Framework](/img/framework.jpg)

## PaRK-Detect Scheme
![PaRK-Detect Scheme](/img/scheme.jpg)

## Graph Optimization Strategy
![GO](/img/graph_optimization_1.jpg, /img/graph_optimization_1.jpg)

- - -

## 1. Code

- - -

## 2. Datasets and Benchmarks

- - -
