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

## PaRK-Detect Scheme
![PaRK-Detect Scheme](/fig/scheme.jpg)

**Illustration of PaRK-Detect scheme.** <br />
**Left:** blue patches contain road while white patches are non-road, black dots are road keypoints, and green lines represent links. <br />
**Right:** the reference point of relative offset is the upper left corner of a patch. Dark yellow patches are linked with the center patch while light yellow ones are not. <br />
We order the eight adjacent patches into numbers 0-7. Here the linked patches are 2, 6, and 7.

## Framework
![Framework](/fig/framework.jpg)

**Overview of our proposed multi-task framework architecture.** <br />
The rectangles are feature maps of different scales. <br />
**I:** input satellite image, <br />
**P:** patch-wise road probability, yellow patches represent non-road while white patches represent road, <br />
**S:** patch-wise road keypoint position, <br />
**L:** patch-wise link status, <br />
**G:** road graph, <br />
**M:** road segmentation mask. <br />
Here we just show 32^2 patches out of 64^2 for better presentation.

## Graph Optimization Strategy
![GO](/fig/graph_optimization.jpg#pic_center)

**Illustration of graph optimization strategies.** <br />
**Left:** connecting adjacent but unconnected endpoints. Red solid lines are links added while red dotted lines are links that should not be added. <br />
**Right:** removing triangle and quadrilateral. Red dotted lines are links removed.

- - -

## 1. Code

### Preprocess

### Train

- - -

## 2. Datasets and Benchmarks

- - -
