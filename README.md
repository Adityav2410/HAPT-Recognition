# HAPT-Recognition
Human Activities and Postural Transitions’ Recognition using Smartphone Data

## PROBLEM STATEMENT DESCRIPTION
Human activities are monitored with the help of Smartphone sensors(Acclerometer and Gyroscope). The statement is to classify the human activities into one of 12 classes based on these sensor readings. 

![alt text](https://github.com/Adityav2410/HAPT-Recognition/blob/master/assets/images/humanActivities.png =250x250) 



## DATASET
[Smartphone-Based Recognition of Human Activities and Postural Transitions Data Set](https://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions "UCI Machine Learning Repository")

The smartphone sensor data are transformed into two categories:- 
*  Time Domain Features - Acclearation(x,y,x), min, median, entropy, etc. 

*  Frequency Domain Features - DFT of time domain features(accleration, jerk magnitude, gyroscope magnitude, etc).


### Data Visualization 
Data is visualized using 2-D PCA and TSNE embeddings. TSNE visualization shows that the different classes are well seperable. 

![alt text][data_viz]

[data_viz]: https://github.com/Adityav2410/HAPT-Recognition/blob/master/assets/images/dataVisualization.png "PCA vs TSNE" = 250x250


## EXPERIMENTS

Several classification techniques are implemented across different parameter variation. A detailed study of all the experiment as mentioned below are presented: 

* Neural Network(Single and Multilayer perceptron)
* SVM(Linear and Gaussian Kernel)
* Boosting(with different loss functions)

### Single Layer Neural Network

| Training Accuracy(%)| Validation Accuracy(%) | Test Accuracy(%) | 
|:-------------------:|-----------------------:| ----------------:|
|        97.55        |        96.2            |       92.13      |



### Multilayer Neural Network

| Number of hidden units|Training Accuracy(%) | Validation Accuracy(%) | Test Accuracy(%) | 
| ----------------------|:-------------------:|-----------------------:| ----------------:|
|          128          |        98.28        |        97.17           |       93.17      |
|          256          |        99.03        |        97.04           |       93.39      |
|          512          |        99.51        |        97.94           |       93.48      |



### L2- SVM


| Kernel   |        Parameters     | Training Accuracy(%)| Validation Accuracy(%) | Test Accuracy(%) | 
| ---------| ----------------------|:-------------------:|-----------------------:| ----------------:|
| Linear   |          C = 1        |        99.53        |        96.98           |       95.19      |
| Gaussian | C = 5000, gamma = 1e-5|        98.83        |        96.6            |       94.4       |
|          5            |        97.72        |        95.57           |       88.7       |



### Boosting

| Loss Function |  Weak learners  | Number of weak learner | Training Accuracy(%)| Validation Accuracy(%)|Test Accuracy(%)| 
| ------------- | --------------- |:----------------------:|--------------------:| ----------------:|---------------------|
| Exponential   | Decision Stumps |          339           |        99.97        |        95.6      |       91.68         |   
| Cross Entropy | Decision Stumps |          303|          |        99.41        |        94.21     |       91.4          |


