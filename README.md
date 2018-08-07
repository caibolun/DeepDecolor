# DeepDecolor: Perception Preserving Decolorization
Bolun Cai, Xiangmin Xu, Xiaofen Xing

### Introduction
Decolorization is a basic tool to transform a color image into a grayscale image, which is used in digital printing, stylized black-and-white photography, and in many single-channel image processing applications. While recent researches focus on retaining as much as possible meaningful visual features and color contrast. In this paper, we explore how to use deep neural networks for decolorization, and propose an optimization approach aiming at perception preserving. The system uses deep representations to extract content information based on human visual perception, and automatically selects suitable grayscale for decolorization. The evaluation experiments show the effectiveness of the proposed method.
![DeepDecolor](http://caibolun.github.io/decolor/framework.jpg)


If you use these codes in your research, please cite:


	@article{cai2018deepdecolor,
		author = {Bolun Cai, Xiangmin Xu and Xiaofen Xing},
		title={Perception Preserving Decolorization},
		journal={IEEE International Conference on Image Processing},
		year={2018}
		}
		
### Install

Here is the list of libraries you need to install to execute the code:
- python = 3.6
- [pytorch](http://pytorch.org/) = 0.4
- torchvision
- numpy
- PIL
- matplotlib
- torchvision
- jupyter

All of them can be installed via `conda` ([anaconda](https://www.anaconda.com/)), e.g.
```
conda install jupyter
```

### Usage

In this repository we provide *Jupyter Notebooks* (`decolor.ipynb`) to reproduce the gray *Impression Sunrise* from the paper.

 - Download the repository and test images
```
git clone https://github.com/caibolun/DeepDecolor.git
```
 - Execute *Jupyter Notebooks* and open `decolor.ipynb` ([Here](https://github.com/caibolun/DeepDecolor/blob/master/decolor.ipynb))
```
jupyter-notebook
```
<img width="400" src="https://raw.githubusercontent.com/caibolun/DeepDecolor/master/images/20.png"/> &nbsp;&nbsp; <img width="400" src="https://raw.githubusercontent.com/caibolun/DeepDecolor/master/gray.png"/>