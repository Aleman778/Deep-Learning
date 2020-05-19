# Exercise 5 - Results

## Task 1
### 1.1 Implement Vanilla GAN
The PyTorch implementation, we used for this lab is from this
[repository](https://github.com/wiseodd/generative-models), small
adjustments were made to run with the original loss function from the
[GAN paper](https://arxiv.org/abs/1406.2661) Ian Goodfellow.

### 1.2 Implementing the logistic loss
This was already done in the provided code repository above.

### 1.3 Running the program for 20K and 100K iterations using both losses
#### Running the original loss from the GAN paper we get the following results
&nbsp; **20K iterations** &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; **100K iterations**

![Original Loss 20K iterations](https://raw.githubusercontent.com/Aleman778/Deep-Learning/master/exercises/exercise5/task1_vanilla_gan_20k.png)
![Original Loss 100K iterations](https://raw.githubusercontent.com/Aleman778/Deep-Learning/master/exercises/exercise5/task1_vanilla_gan_100k.png)

#### Running the logistic loss we get the following results
&nbsp; **20K iterations** &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; **100K iterations**

![Logistic Loss 20K iterations](https://raw.githubusercontent.com/Aleman778/Deep-Learning/master/exercises/exercise5/task1_vanilla_gan_logistic_loss_20k.png)
![Logistic Loss 100K iterations](https://raw.githubusercontent.com/Aleman778/Deep-Learning/master/exercises/exercise5/task1_vanilla_gan_logistic_loss_100k.png)

## Task 2 Train CNN MNIST model to create adversarial images
### 2.1 Classify 4s as 9s
![Adversarial 1](https://raw.githubusercontent.com/Aleman778/Deep-Learning/master/exercises/exercise5/task2_adversarial_1.png)
![Adversarial 2](https://raw.githubusercontent.com/Aleman778/Deep-Learning/master/exercises/exercise5/task2_adversarial_2.png)
![Adversarial 3](https://raw.githubusercontent.com/Aleman778/Deep-Learning/master/exercises/exercise5/task2_adversarial_3.png)
![Adversarial 4](https://raw.githubusercontent.com/Aleman778/Deep-Learning/master/exercises/exercise5/task2_adversarial_4.png)
![Adversarial 5](https://raw.githubusercontent.com/Aleman778/Deep-Learning/master/exercises/exercise5/task2_adversarial_5.png)

### 2.2 Using random noise to classify it as 9
![Adversarial Noise 1](https://raw.githubusercontent.com/Aleman778/Deep-Learning/master/exercises/exercise5/task2_adversarial_noise_1.png)
![Adversarial Noise 2](https://raw.githubusercontent.com/Aleman778/Deep-Learning/master/exercises/exercise5/task2_adversarial_noise_2.png)
![Adversarial Noise 3](https://raw.githubusercontent.com/Aleman778/Deep-Learning/master/exercises/exercise5/task2_adversarial_noise_3.png)
![Adversarial Noise 4](https://raw.githubusercontent.com/Aleman778/Deep-Learning/master/exercises/exercise5/task2_adversarial_noise_4.png)
![Adversarial Noise 5](https://raw.githubusercontent.com/Aleman778/Deep-Learning/master/exercises/exercise5/task2_adversarial_noise_5.png)

### 2.3 Using zeros image to classify it as 9
![Adversarial Zeros 1](https://raw.githubusercontent.com/Aleman778/Deep-Learning/master/exercises/exercise5/task2_adversarial_zeros_1.png)
![Adversarial Zeros 2](https://raw.githubusercontent.com/Aleman778/Deep-Learning/master/exercises/exercise5/task2_adversarial_zeros_2.png)
![Adversarial Zeros 3](https://raw.githubusercontent.com/Aleman778/Deep-Learning/master/exercises/exercise5/task2_adversarial_zeros_3.png)
![Adversarial Zeros 4](https://raw.githubusercontent.com/Aleman778/Deep-Learning/master/exercises/exercise5/task2_adversarial_zeros_4.png)
![Adversarial Zeros 5](https://raw.githubusercontent.com/Aleman778/Deep-Learning/master/exercises/exercise5/task2_adversarial_zeros_5.png)

## Task 3 DCGAN on CelebA dataset (from project work)
For fun we ran our DCGAN code from the project work on the CelebA dataset:
![DCGAN on CelebA](https://raw.githubusercontent.com/Aleman778/Deep-Learning/master/exercises/exercise5/task3_dcgan_celeba.png)
