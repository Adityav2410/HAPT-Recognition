load('/Users/shashank/Documents/Quarter 2/Statistical Learning II/Project/HAPT Data Set/Train/train.mat');
load('./Train/y_train.txt')
[p,~,~,~,exp,~] = pca(Xtrain);
r_Xtrain = Xtrain*p(:,1:3);
scatter(r_Xtrain(:,1),r_Xtrain(:,2),[],y_train);
figure
plot(cumsum(exp))

%figure
%scatter3(r_Xtrain(:,1),r_Xtrain(:,2),r_Xtrain(:,3))