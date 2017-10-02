from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import decomposition
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace as bp
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV

print 'Loading data...'
X_tr = np.loadtxt('../HAPT Data Set/Train/X_train.txt', delimiter=' ')
Y_tr = np.loadtxt('../HAPT Data Set/Train/y_train.txt')
train_size = int(X_tr.shape[0]*0.8)
X_train = X_tr[0:train_size]
Y_train = Y_tr[0:train_size]
X_val = X_tr[train_size:]
Y_val = Y_tr[train_size:]

X_test = np.loadtxt('../HAPT Data Set/Test/X_test.txt', delimiter=' ')
Y_test = np.loadtxt('../HAPT Data Set/Test/y_test.txt')
print 'Done!'


#### Naive Bayes Classifier ####

print 'Training Naive Bayes Classifier...'
gnb = GaussianNB()
nb_clf = gnb.fit(X_train,Y_train)
print 'Done!'
print 'Calculating accuracy...'
pred = nb_clf.predict(X_test)
accuracy = np.sum(pred==Y_test)/float(X_test.shape[0])
print 'Naive Bayes Accuracy: %.4f' % accuracy


#### Linear SVM Classifier ####
# line search is done for the optimum value of C
# Note: sklearn's inbuilt methods can also be used for this purpose

indx=0
accuracy = np.zeros(11)
for i in range(-5,5):
        print 'Training linear svm... for C:{}'.format(2**i)
        lsvm_clf = svm.SVC(C=2**i, kernel='linear')
        lsvm_clf.fit(X_train,Y_train)
        print 'Done!'

        print 'Calculating accuracy...'
        pred = lsvm_clf.predict(X_train)
        acc = np.sum(pred==Y_train)/float(X_train.shape[0])
        print 'Linear SVM Train Accuracy: %.4f' % acc

        pred = lsvm_clf.predict(X_val)
        accuracy[indx] = np.sum(pred==Y_val)/float(X_val.shape[0])
        print 'Linear SVM Validation Accuracy for C=%f: %.4f' % (2**i,accuracy[indx])
        indx += 1
plt.figure()
plt.plot(accuracy)
plt.tick_params(labelright = True)
plt.title('Validation accuracy vs C for linear SVM')



#### Gaussian Kernel SVM Classifier ####
# grid search is done for the optimum value of C and gamma

print 'Training Gaussian SVM'
C_range = np.logspace(-5, 5, 10)
gamma_range = np.logspace(-5, -5, 10)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
grid.fit(X_test, Y_test)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))


#### Polynomial kernel SVM classifier ####

print 'Training polynomial kernel svm...'
gsvm_clf = svm.SVC(kernel='poly',C = 1, gamma=1e-5)
gsvm_clf.fit(X_train,Y_train)
print 'Done!'
print 'Calculating accuracy...'
pred = gsvm_clf.predict(X_train)
accuracy = np.sum(pred==Y_train)/float(X_train.shape[0])
print 'Polynomial kernel SVM Accuracy: %.4f' % accuracy


#### Dimensionality reduction using PCS and analysis using kernel SVM ####
pca = decomposition.PCA()
pca.fit(X_tr)
accuracy = np.zeros(12)
indx = 0
for n_components in xrange(10,561,50):
        print n_components
        # print '% variance explained: ', 100*np.cumsum(pca.explained_variance_ratio_)[n_components-1]
        X_red = np.dot(X_tr,pca.components_[:,0:n_components])

        # print 'Training linear svm...'
        lsvm_clf = svm.SVC(C=1, kernel='linear')
        lsvm_clf.fit(X_red,Y_tr)
        # print 'Done!'

        pred = lsvm_clf.predict(np.dot(X_test,pca.components_[:,0:n_components]))
        accuracy[indx] = np.sum(pred==Y_test)/float(X_test.shape[0])
        # print 'Linear SVM Test Accuracy: %.4f' % accuracy[indx]
        indx += 1

plt.figure()
plt.plot(np.arange(10,561,50),accuracy)
plt.grid()
plt.ylabel('Accuracy (%)')
plt.xlabel('Number of features')
plt.savefig('acc_vs_features.eps', format='eps', dpi=1000)
plt.show()

bp()





