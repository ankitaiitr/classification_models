from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import data_prep
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score
def random_forest(X_train,y_train):
    param_grid = {'max_depth': np.arange(3,10)}
    clf = RandomForestClassifier(n_estimators= 500,criterion='entropy',random_state=0)
    clf = GridSearchCV(clf, param_grid)
    clf.fit(X_train, y_train)
    #print("Feature Importances....")
    #print(clf.feature_importances_)
    print(".........................")
    return clf

def test_clf(clf,X_test,y_test):
    pred= clf.predict(X_test)
    print('Accuracy: '+str(sum(pred==y_test)/len(y_test)))
    #precision, recall, _ = precision_recall_curve(y_test,pred)
    #average_precision = average_precision_score(y_test,pred)
    #print("precision "+str(precision)+"recall "+str(recall))
    print("Recall" +str(recall_score(y_test, pred, average=None)))
    return confusion_matrix(y_test, pred)
    
def dt(X_train,y_train):
    from sklearn import tree
    param_grid = {'max_depth': np.arange(3, 25)}
    clf = tree.DecisionTreeClassifier(random_state=0)
    tree = GridSearchCV(clf, param_grid)
    
    clf = tree.fit(X_train,y_train)
    return clf

def ada(X_train,y_train):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    clf = AdaBoostClassifier(DecisionTreeClassifier(random_state=0,max_depth=10),
                         algorithm="SAMME",
                         n_estimators=1000,random_state=0 )
    clf.fit(X_train, y_train)
    return clf

def svc(X_train,y_train):
    from sklearn import svm
    clf = svm.SVC(kernel = 'rbf')
    clf = clf.fit(X_train,y_train)
    return clf
def mlp(X_train,y_train):
    from sklearn.neural_network import MLPClassifier
def main(enc='le'):
    X_train,X_test,y_train,y_test = data_prep.load_data()
    data_prep.describe_df(pd.DataFrame(X_train))
    if enc == 'ohe':
        X_train = data_prep.ohe_encode(X_train,np.delete(list(range(0,X_train.shape[1])),[1,4,12]))
        ohe = pickle.load(open('../res/ohe.pkl','rb'))
        X_test=ohe.transform(X_test)
    

# =============================================================================
# No feature engineering 
# =============================================================================
    print("Random Forest.....")
    clf = random_forest(X_train,y_train)
    print(test_clf(clf,X_test,y_test))
    print("Decision Tree.....")
    clf = dt(X_train,y_train)
    print(test_clf(clf,X_test,y_test))
    print("SVC.....")
    clf = svc(X_train,y_train)
    print(test_clf(clf,X_test,y_test))
    print("Adaboost.....")
    clf = ada(X_train,y_train)
    print(test_clf(clf,X_test,y_test))

# =============================================================================
# PCA 
# =============================================================================
    X_train_pca = data_prep.feature_selection_pca(X_train,17)
    pca = pickle.load(open('../res/pca.pkl','rb'))
    X_test_pca = pca.transform(X_test)
    print("With PCA")
    print("Random Forest.....")
    clf = random_forest(X_train_pca,y_train)
    print(test_clf(clf,X_test_pca,y_test))
    print("Decision Tree.....")
    clf = dt(X_train_pca,y_train)
    print(test_clf(clf,X_test_pca,y_test))
    print("SVC.....")
    clf = svc(X_train_pca,y_train)
    print(test_clf(clf,X_test_pca,y_test))
    print("Adaboost.....")
    clf = ada(X_train_pca,y_train)
    print(test_clf(clf,X_test_pca,y_test))
# =============================================================================
# Select K features 
# =============================================================================
    if enc=="le":
        np.random.seed(9)
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
        feat = data_prep.feature_selection(X_train,y_train,15)
        X_train_feat = X_train[feat]
        X_test_feat = X_test[feat]
        print("With Features selector")
        print("Random Forest.....")
        clf = random_forest(X_train_feat,y_train)
        print(test_clf(clf,X_test_feat,y_test))
        print("Decision Tree.....")
        clf = dt(X_train_feat,y_train)
        print(test_clf(clf,X_test_feat,y_test))
        print("SVC.....")
        clf = svc(X_train_feat,y_train)
        print(test_clf(clf,X_test_feat,y_test))
        print("Adaboost.....")
        clf = ada(X_train_feat,y_train)
        print(test_clf(clf,X_test_feat,y_test))
# =============================================================================
#OHE
# =============================================================================
main()
main(enc = 'ohe')