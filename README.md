# HAPT-Recognition
Human Activities and Postural Transitions’ Recognition using Smartphone Data

## PROBLEM STATEMENT DESCRIPTION
Human activities are monitored with the help of Smartphone sensors(Acclerometer and Gyroscope). The statement is to classify the human activities into one of 12 classes based on these sensor readings. 

![alt text][humanActivity]

[humanActivity]: https://github.com/Adityav2410/HAPT-Recognition/blob/master/assets/images/humanActivities.png "HUMAN ACTIVITIES"





## DATASET
[Smartphone-Based Recognition of Human Activities and Postural Transitions Data Set](https://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions "UCI Machine Learning Repository")

The smartphone sensor data are transformed into two categories:- 
*  Time Domain Features - Acclearation(x,y,x), min, median, entropy, etc. 

*  Frequency Domain Features - DFT of time domain features(accleration, jerk magnitude, gyroscope magnitude, etc).


### Data Visualization 
Data is visualized using 2-D PCA and TSNE embeddings. TSNE visualization shows that the different classes are well seperable. 

![alt text][data_viz]

[data_viz]: https://github.com/Adityav2410/HAPT-Recognition/blob/master/assets/images/dataVisualization.png "PCA vs TSNE"


## EXPERIMENTS

Several classification techniques are implemented across different parameter variation. A detailed study of all the experiment as mentioned below are presented: 

* K nearest neighbor
* Neural Network(Single and Multilayer perceptron)
* SVM(Linear and Gaussian Kernel)
* Boosting(with different loss functions)




