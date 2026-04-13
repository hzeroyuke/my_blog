
Point Cloud和Splatting是在计算机中构建3d表示的方案

## 1. 3d RePresentation

在计算机中表征3d的方式有很多种，并没有像2d的图片表示那么统一

### 1.1. PointCloud

Point Cloud是一种较为传统的3D表征的方案，就是一堆带有三维坐标和RGB数值的点

Point Cloud往往通过LiDAR扫描和RGBD深度相机扫描获得

其需要经过重建之后才能真实渲染出3d场景，并且容易丢失精度

### 1.2. 3D Gaussian Splatting

3d Gaussian Splatting又称3DGS，是一种更加现代的3d表示方案

3DGS的单位不是点，而是高斯椭球，每个球有

- μ（中心位置）、Σ（协方差，决定形状和朝向）
- α（不透明度）
- 球谐系数（spherical harmonics，用来表示随视角变化的颜色）

基于上述信息，3dGS能够表征的信息远比PointCloud丰富，并且能够做到实时渲染，现阶段很多的3d构建都是先有点云，再通过这个点云来构建3dGS

基于点云构建3d GS的方案也是梯度下降，但是其内部却不是对神经网络进行梯度下降，而是对这些椭球的参数进行梯度下降地更新

### 1.3. Mesh


### 1.4. Voxel

### 1.5. NeRF


## 2. 3d Generation

现代的3d生成均以Mesh作为生成目标，因为这是可以在各种工业软件和游戏软件中直接使用的3d表述

而 Point Cloud，3D Gaussian Splatting 这些往往作为中间产物存在

