from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import data_prep
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import recall_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
classifier_idx=['SVC','DT','RF','MLP','ADA','NB']
classifiers = [
    SVC(gamma=1, C=0.001),
    DecisionTreeClassifier(max_depth=7,random_state=0),
    RandomForestClassifier(n_estimators= 1000,criterion='entropy',random_state=0),
    MLPClassifier(activation='relu',learning_rate='adaptive'),
    AdaBoostClassifier(DecisionTreeClassifier(random_state=0,max_depth=10),
                         algorithm="SAMME",
                         n_estimators=1000,random_state=0 ),
    GaussianNB()]

X_train,X_test,y_train,y_test = data_prep.load_data()
#data_prep.describe_df(pd.DataFrame(X_train))

def process_clf(X_train,y_train,X_test,y_test,filename):
    record = pd.DataFrame()
    for count,clf in enumerate(classifiers):
        print('Working on '+classifier_idx[count])
        clf = classifiers[count]
        clf= clf.fit(X_train,y_train)
        pred= clf.predict(X_test)
        acc = sum(pred==y_test)/len(y_test)
        recall = recall_score(y_test, pred, average=None)
        rec = pd.DataFrame({'accuracy':acc,'recall':recall,'confusion_mat':str(confusion_matrix(y_test, pred)),'classifier':classifier_idx[count]})
        record= record.append(rec)
    
    record.to_csv(filename)
    return record

def run_pca(X_train,y_train,X_test,y_test):
    X_train_pca = data_prep.feature_selection_pca(X_train,17)
    pca = pickle.load(open('../res/pca.pkl','rb'))
    X_test_pca = pca.transform(X_test)
    return process_clf(X_train_pca,y_train,X_test_pca,y_test,'../out/All_clf_records_pca.csv')    

def percentile_feat(X_train,y_train,X_test,y_test):
    np.random.seed(9)
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    feat = data_prep.feature_selection(X_train,y_train,15)
    X_train_feat = X_train[feat]
    X_test_feat = X_test[feat]
    return process_clf(X_train_feat,y_train,X_test_feat,y_test,'../out/All_clf_records_feature.csv')    

def generate_graph(record,figname):
    import matplotlib.pyplot as plt
    record_0 = record[record.index==0]
    record_0 = record_0.set_index('classifier')
    fig = record_0.plot( y=["recall", "accuracy"], kind="bar").get_figure()
    fig.savefig(figname)
    plt.close(fig)
# =============================================================================
# run codes for Label encoders
# =============================================================================
record = process_clf(X_train,y_train,X_test,y_test,'../out/All_clf_records_nofeature_selection.csv')    
generate_graph(record,'../out/le_nofeat.png')
record_pca = run_pca(X_train,y_train,X_test,y_test)
generate_graph(record_pca,'../out/le_pca.png')
record_percentile = percentile_feat(X_train,y_train,X_test,y_test)
generate_graph(record_percentile,'../out/le_percentile.png')
# =============================================================================
# run codes for ohe
# =============================================================================
X_train = data_prep.ohe_encode(X_train,np.delete(list(range(0,X_train.shape[1])),[1,4,12]))
ohe = pickle.load(open('../res/ohe.pkl','rb'))
X_test=ohe.transform(X_test)

record_ohe = process_clf(X_train,y_train,X_test,y_test,'../out/oheAll_clf_records_nofeature_selection.csv')    
generate_graph(record_ohe,'../out/ohe_nofeat.png')
record_pca_ohe = run_pca(X_train,y_train,X_test,y_test)
generate_graph(record_pca_ohe,'../out/ohe_pca.png')