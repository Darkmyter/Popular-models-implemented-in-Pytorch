# <center>Pytroch implementation of popular models</center>

Implementation and light training examples[^1] for popular Deep Learning models in Pytorch from scratch.  

[^1]: PS: The current goal of this project is to present simple examples written from scratch. It doesn't dive deep into optimizing the performance or analyzing the training (At least for now).

## Computer Vision
### Image classification
* [AlexNet](./Computer-vision/image-classification/alexnet-cifar10.ipynb): trained on CIFAR-10, [paper](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
* [VGG-16](Computer-vision/image-classification/vgg16-covid-19.ipynb): trained on Covid-19 chest x-ray dataset from [kaggle](https://www.kaggle.com/datasets/francismon/curated-covid19-chest-xray-dataset), [paper](https://arxiv.org/pdf/1409.1556.pdf)
* [InceptionV1 (GoogLeNet)](Computer-vision/image-classification/inceptionv1-brain-tumor.ipynb): trained on brain tumor MRI dataset from [kaggle](https://www.kaggle.com/datasets/preetviradiya/.brian-tumor-dataset), [paper](https://arxiv.org/pdf/1409.4842v1.pdf)
* [ResNet-50](./Computer-vision/image-classification/resnet-50-animal-10.ipynb): trained on Animal-10 dataset from [kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10), [paper](https://arxiv.org/abs/1512.03385)
* [EfficientNet](./Computer-vision/image-classification/efficient-net-car-damage.ipynb): trained  on car damage severity from [kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10), [paper](https://arxiv.org/pdf/1905.11946.pdf)
* [Vision Transformer](./Computer-vision/image-classification/vit-human-action-recognition.ipynb): trained on human action recognition from [kaggle](https://www.kaggle.com/datasets/shashankrapolu/human-action-recognition-dataset), [paper](https://arxiv.org/abs/2010.11929)