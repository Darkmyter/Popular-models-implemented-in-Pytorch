# <center>Pytroch implementation of popular models</center>

Implementation and light training examples[^1] for popular Deep Learning models in Pytorch from scratch.  

[^1]: PS: The current goal of this project is to present simple examples written from scratch. It doesn't dive deep into optimizing the performance or analyzing the training (At least for now).

## Computer Vision
### Image classification
* [AlexNet](./Computer-vision/image-classification/alexnet-cifar10.ipynb): [paper](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf), trained on CIFAR-10
* [VGG-16](Computer-vision/image-classification/vgg16-covid-19.ipynb): [paper](https://arxiv.org/pdf/1409.1556.pdf), trained on Covid-19 chest x-ray dataset from [kaggle](https://www.kaggle.com/datasets/francismon/curated-covid19-chest-xray-dataset)
* [InceptionV1 (GoogLeNet)](Computer-vision/image-classification/inceptionv1-brain-tumor.ipynb): [paper](https://arxiv.org/pdf/1409.4842v1.pdf), trained on brain tumor MRI dataset from [kaggle](https://www.kaggle.com/datasets/preetviradiya/.brian-tumor-dataset)
* [ResNet-50](./Computer-vision/image-classification/resnet-50-animal-10.ipynb): [paper](https://arxiv.org/abs/1512.03385), trained on Animal-10 dataset from [kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10)