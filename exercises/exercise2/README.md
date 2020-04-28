# Exercise 2 - Results

## Part 1 - Transfer Learning from ImageNet
- Training AlexNet on CIFAR10 dataset (with seed 42) results in the
  accuracy **14.040%** on the test set.
  
![Confusion Matrix](https://raw.githubusercontent.com/Aleman778/Deep-Learning/master/exercises/exercise2/part1_confusion_matrix.png)

- Training AlexNet on CIFAR10 dataset (with seed 42) using a
  pretrained model on ImageNet results in the accuracy **86.520%** on the
  test set.
  
  
![Confusion Matrix](https://raw.githubusercontent.com/Aleman778/Deep-Learning/master/exercises/exercise2/part2_confusion_matrix.png)

- The difference between the first and the second experiments is that
  in the second one we use a pretrained model on ImageNet that has
  been trained for weeks. The reason why we get higher test
  accuracy on the second experiment is because the pretrained model
  have learnt many features that we can apply to when training on
  CIFAR10 dataset. Also the pretrained model has been trained for
  many more epoches than the model used in the first experiment.

## Part 2 - Transfer Learning from MNIST
- Training CNN_basic on SVHN dataset (with seed 42) results in the
  accuracy **14.909%** on the test set.
  
![Confusion Matrix](https://raw.githubusercontent.com/Aleman778/Deep-Learning/master/exercises/exercise2/part3_confusion_matrix.png)

- Training CNN_basic on MNIST dataset (with seed 42) results in the
  accuracy **88.390%** on the test set.
  
![Confusion Matrix](https://raw.githubusercontent.com/Aleman778/Deep-Learning/master/exercises/exercise2/part4_confusion_matrix.png)


- Training CNN_basic on SVHN dataset (with seed 42) on the previously
  trained model above we get accuracy **28.834%** on the test set.
  
![Confusion Matrix](https://raw.githubusercontent.com/Aleman778/Deep-Learning/master/exercises/exercise2/part5_confusion_matrix.png)
