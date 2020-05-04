# Exercise 3 - Results

## Model 1: Almost Untrained 
- Training CNN_basic on MNIST with learning rate 1e-6 and for one epoch only results in **6.470%** accuracy on the test set.

- PCA
![PCA](https://raw.githubusercontent.com/Aleman778/Deep-Learning/master/exercises/exercise3/PCA1.png)

- t-SNE
![tSNE](https://raw.githubusercontent.com/Aleman778/Deep-Learning/master/exercises/exercise3/tSNE1.png)

## Model 2: Long Training
- Training CNN_basic on MNIST for 10 epochs results in **90.970%** accuracy on the test set.

- PCA
![PCA](https://raw.githubusercontent.com/Aleman778/Deep-Learning/master/exercises/exercise3/PCA10.png)

-t-SNE
![tSNE](https://raw.githubusercontent.com/Aleman778/Deep-Learning/master/exercises/exercise3/tSNE10.png)


## Part A - PCA vs t-SNE
PCA is mainly concerned in preserving large distances. Otherwise, it is not good in reproducing local structure from high dimensional space to the low dimensional map. Comparing PCA1.png (training for 1 epoch) and PCA10.png (training for 10 epochs), we do not see big differences, although the accuracy has jumped from 6 % to 90%!

t-SNE is good in reproducing structure of data: it keeps similar points close to each other in the low dimension space, and unsimilar ones far apart each other. We can see almost 10 distiguishable clusters in the final figure tSNE10.png.

## Part B - Short vs Long Training
The test accuracy changes from 6% in the first model to 90% in the second one. Training is crucial to get high test accuracy, provided there is no overfitting. Comparing tSNE1.png (first model) with tSNE10.png (second model), in the first one there is no clear structure, whereas in the second one, we can see 10 distinct clusters. If we could color the clusters as they did in the video lecture about t-SNE, that distinction would be clearer and we would know which digit belonged to which cluster.

