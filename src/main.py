import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import zero_one_loss


def normalize(x_t):
        f_params = {}
        for i in range(x_t.shape[1]):
            f_mean = np.mean(x_t[:, i])
            f_std = np.std(x_t[:, i])
            f_params[i] = [f_mean, f_std]
        nrm_data = np.zeros((61, 2), dtype=object)
        new_data = np.zeros_like(x_t)
        for i in range(x_t.shape[1]):
            f_mean, f_std = f_params[i][0], f_params[i][1]
            new_data[:, i] = (x_t[:, i]-f_mean)/f_std
            nrm_data[i, 0], nrm_data[i,1] = "%.5f" % np.mean(new_data[:, i]), "%.5f" % np.var(new_data[:, i])
        return new_data

#loading data and setting columns
data=pd.read_csv('VSTrain.dt')
data_test = pd.read_csv('VSTest.dt')

data.columns=['amplitude','beyondlstd','flux_percentile_ratio_mid20','flux_percentile_ratio_mid35','flux_percentile_ratio_mid50','flux_percentile_ratio_mid65','flux_percentile_ratio_mid80','fold2P_slope_10percentile','fold2P_slope_90percentile','freql_harmonics_amplitude_0','freql_harmonics_amplitude_1','freql_harmonics_amplitude_2','freql_harmonics_amplitude_3','freql_harmonics_freq_0','freql_harmonics_rel_phase_1','freql_harmonics_rel_phase_2','freql_harmonics_rel_phase_3','freq2_harmonics_amplitude_0','freq2_harmonics_amplitude_1','freq2_harmonics_amplitude_2','freq2_harmonics_amplitude_3','freq2_harmonics_freq_0','freq2_harmonics_rel_phase_1','freq2_harmonics_rel_phase_2','freq2_harmonics_rel_phase_3','freq3_harmonics_amplitude_0','freq3_harmonics_amplitude_1','freq3_harmonics_amplitude_2','freq3_harmonics_amplitude_3','freq3_harmonics_freq_0','freq3_harmonics_rel_phase_1','freq3_harmonics_rel_phase_2','freq3_harmonics_rel_phase_3','freq_amplitude_ratio_21','freq_amplitude_ratio_31','freq_frequency_ratio_21','freq_frequency_ratio_31','freq_signif','freq_signif_ratio_21','freq_signif_ratio_21','freq_varrat','freq_y_offset','linear_trend','max_slope','median_absolute_deviation','median_buffer_range_percentage','medperc90_2p_p','p2p_scatter_2praw','p2p_scatter_over_mad','p2p_scatter_pfold_over_mad','p2p_ssqr_diff_over_var','percent_amplitude','percent_difference_flux_percentile','QSO','non_QSO','scatter_res_raw','skew','small_kurtosis','std','stetson_j','stetson_k','class_number']
data_test.columns = data.columns

#to plot the class frequencies
#print(data['class_number'].value_counts(normalize="True"))
a = data['class_number'].value_counts(normalize="True")
ax = a.plot.bar(x='class_number', y='val', rot=0)
ax.legend(loc=1)  

#to drop the classes with inadequate number of classes (preprocessing step 1)
b = data['class_number'].value_counts()[data['class_number'].value_counts()<65].index
c = data[data['class_number'].isin(b)].index
d = data_test[data_test["class_number"].isin(b)].index
data.drop(c, inplace=True)
data_test.drop(d, inplace=True)
#print(data)
#print(data_test)

x=data.iloc[:,:-1].values
y=data.iloc[:,-1]
x_test = data_test.iloc[:,:-1].values
y_test =data_test.iloc[:,-1]

#to normalize the data as a preporcessing step (preprocessing step 2)
x_train_normalized = normalize(x)
print(x_train_normalized.shape)

#pricipal component analysis
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(x_train_normalized)
print(X_train_pca.shape)
principalDf = pd.DataFrame(data = X_train_pca
             , columns = ['principal_component_1', 'principal_component_2'])

y = y.reset_index(drop=True)
finalDf = pd.concat([principalDf, y], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0,3,22,23]
colors = ['r', 'g', 'b','y']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['class_number'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal_component_1']
               , finalDf.loc[indicesToKeep, 'principal_component_2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

#4-means clustering of the data
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42).fit(principalDf)

c = kmeans.cluster_centers_

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Cluster center', 0,3,22,23]
ax.scatter(c[:, 0], c[:, 1], s = 130, c = 'black', label = 'Centroids', zorder=3)
colors = ['y','r', 'g', 'b','y']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['class_number'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal_component_1']
               , finalDf.loc[indicesToKeep, 'principal_component_2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

#multi-nomial logistic regression
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=6000)
model.fit(x, y)
train_preds = model.predict(x)
loss = zero_one_loss(y, train_preds)
print("train set loss: %.4f" %loss)

test_preds = model.predict (x_test)
loss = zero_one_loss(y_test, test_preds)
print("test set loss: %.4f" %loss)

#random forest classificator
model = RandomForestClassifier(n_estimators=20, criterion="entropy",max_features="sqrt",oob_score=True)
model.fit(x, y)
print("oob score: %.4f" %model.oob_score_)
train_preds = model.predict(x)
loss = zero_one_loss(y, train_preds)
print("train set loss: %.4f" %loss)

test_preds = model.predict (x_test)
loss = zero_one_loss(y_test, test_preds)
print("test set loss: %.4f" %loss)

model = RandomForestClassifier(n_estimators=20, criterion="entropy",max_features=None, oob_score=True)
model.fit(x, y)
print("oob score: %.4f" %model.oob_score_)
train_preds = model.predict(x)
loss = zero_one_loss(y, train_preds)
print("train set loss: %.4f" %loss)

test_preds = model.predict (x_test)
loss = zero_one_loss(y_test, test_preds)
print("test set loss: %.4f" %loss)

#k-nearest neighbours classification

#cross_validation implementation to decide on the number of neighbours
def cross_validation(x,y):
    
    n_nei = list(range(1, 19)) #chosen due to practice of sqrt of number of training samples being 17 (close)
    lowest_loss = None
    lowest_loss_par = []
    kfolds = list(StratifiedKFold(n_splits=5, shuffle=True).split(x, y))
    losses_for_params = []
    for n in n_nei:
        for train_index, test_index in kfolds:
            x_train_fold, y_train_fold = x[train_index], y[train_index]
            x_test_fold, y_test_fold  = x[test_index], y[test_index]
            model = KNeighborsClassifier(n_neighbors=n)
            model.fit(x_train_fold, y_train_fold)
            preds_fold = model.predict(x_test_fold)
            loss = zero_one_loss(y_test_fold, preds_fold)
            losses_for_params.append(loss)
        loss = np.mean(losses_for_params)
        if lowest_loss is None or loss < lowest_loss:
            lowest_loss, lowest_loss_par = loss, n
    print("lowest found mean loss: %s. n_neighbors = %s" % (lowest_loss, lowest_loss_par))
    return lowest_loss_par

n_ne = cross_validation(x,y)
knn = KNeighborsClassifier(n_neighbors=n_ne)
knn.fit(x, y)
train_preds = knn.predict(x)
loss = zero_one_loss(y, train_preds)
print("train set loss: %.4f" %loss)

test_preds = knn.predict (x_test)
loss = zero_one_loss(y_test, test_preds)
print("test set loss: %.4f" %loss)