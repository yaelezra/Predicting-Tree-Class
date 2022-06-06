# %% importing

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
from collections import Counter

from skimage.feature import hog
from skimage.transform import resize

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

from project_functions import hist_wh, class2mat, train_test_model

# %%  defining path of data and listing images
# insert the path in which you saved the images
images_path = 'C:/Users/Yael A/Desktop/machine learning for neuroscience/project/project data'
images_names = os.listdir(images_path)

# %% hist of widths and heights

widths = []
heights = []

for i in range(0, len(images_names)):
    image_name = images_names[i]
    image_path = images_path + '/' + image_name
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    widths.append(img.shape[1])
    heights.append(img.shape[0])

plt.hist(widths)
plt.xlabel('width')
plt.ylabel('count')
plt.title('histogram of widths')
plt.show()

plt.hist(heights)
plt.xlabel('height')
plt.ylabel('count')
plt.title('histogram of heights')
plt.show()

# choosing new width and new height for resizing
new_width = hist_wh(widths)
new_height = hist_wh(heights)

# %% intra class variability

class_sizes = {}
for c in range(0, 15):
    # saving the resized images of this class as a mat
    subs = 'l' + str(c + 1) + 'nr'
    class_mat = class2mat(subs, new_width, new_height, images_names, images_path)
    class_shape = class_mat.shape
    class_sizes[subs] = class_shape[2]

    # plotting image of intra class sd
    var_img = np.std(class_mat, axis=2)
    plt.imshow(var_img, cmap='gray')
    plt.title(subs)
    plt.show()

# %% reading the images and using hog

features = []
all_labels = []

for i in range(0, len(images_names)):
    # reading the image
    image_name = images_names[i]
    image_path = images_path + '/' + image_name
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # resizing
    img_resized = resize(img, (new_height, new_width))

    # feature extraction using hog
    pix = 300
    hog_features, hog_img = hog(
        img_resized, pixels_per_cell=(pix, pix),
        cells_per_block=(2, 2),
        orientations=9,
        visualize=True,
        block_norm='L2-Hys')

    features.append(np.transpose(hog_features))

    # saving label
    label = image_name[1:image_name.find('nr')]
    all_labels.append(int(label))

features_data = np.array(features)

# %% exploring features data

features_df = pd.DataFrame(features_data)
des_features = features_df.describe()

# %% PCA

# mean center
data_centered = preprocessing.scale(features_data, with_std=False, axis=1)

# fitting PCA
pca = PCA()
pca.fit(data_centered)

# plotting explained variance
exp_var = pca.explained_variance_ratio_
exp_var_cumsum = exp_var.cumsum()
plt.plot(exp_var_cumsum, label='Cumulative sum')
plt.plot(exp_var, label='Individual component')

plt.title("Explained Variance by Number of Principal Components")
plt.xlabel('# Principal component')
plt.ylabel("Explained Variance")
plt.legend()
plt.show()

# not good enough for EDA

# %% reducing dimensions to 100 dimensions

pca100 = PCA(n_components=100)
pca100.fit(data_centered)
data_reduced = pca100.transform(data_centered)
reduced_var = pca100.explained_variance_ratio_.cumsum()

# %% PCA for each class

explained_variance = np.empty((15, 75))
labels_array = np.array(all_labels)
for i in range(0, 14):
    class_label = i + 1
    label_idxs = np.where(labels_array == class_label)
    PCA_data = data_centered[label_idxs]
    class_pca = PCA()
    class_pca.fit(PCA_data)
    plt.plot(class_pca.explained_variance_ratio_.cumsum(),
             label='class' + str(class_label))

plt.title("Explained Variance by Number of Principal Components")
plt.xlabel('# Principal component')
plt.ylabel("Explained Variance")
plt.legend()
plt.show()

# %% tSNE

# fitting tSNE
tSNE = TSNE()
tSNE_features = tSNE.fit_transform(features_data)

tSNE_df = pd.DataFrame(tSNE_features)
tSNE_df.columns = ['x', 'y']
tSNE_df['class'] = labels_array

# visualizing
sns.scatterplot(x='x', y='y', hue='class', data=tSNE_df,
                palette=sns.color_palette(cc.glasbey, n_colors=15))
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0)
plt.title('tSNE')
plt.show()

# %% labels - choosing thw wanted class to predict

wanted_label = 7
labels = labels_array == wanted_label
labels = np.dot(labels, 1)

# %% train test split

X_train, X_test, y_train, y_test = train_test_split(data_reduced, labels, test_size=0.33, random_state=0)

# %% Showing imbalanced data - example of 2 features

counter_imbalanced = Counter(labels)
sns.scatterplot(X_train[:, 49], X_train[:, 99], hue=y_train)
plt.xlabel('column 50')
plt.ylabel('column 100')
plt.title(counter_imbalanced)
plt.show()

# %% MlP classifier

model_MLP = MLPClassifier(random_state=0)

MLP_parameters = {
    'model__hidden_layer_sizes': [(5,), (50,), (100,), (5, 5), (5, 100), (100, 100)],
    'model__activation': ['identity', 'logistic', 'tanh', 'relu'],
    'model__alpha': [0.0001, 0.05, 0.5],
    'model__learning_rate': ['constant', 'invscaling', 'adaptive'],
    'model__learning_rate_init': [0.001, 0.1, 10]
}

MLP_params, MLP_train, MLP_validation = train_test_model(model_MLP, MLP_parameters,
                                                         X_train, y_train,
                                                         X_test, y_test)

# %% SVM

model_SVM = SVC(random_state=0, gamma='auto', probability=True)

SVM_parameters = {
    'model__C': [0.05, 0.5, 1, 10, 100],
    'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

SVM_params, SVM_train, SVM_validation = train_test_model(model_SVM, SVM_parameters,
                                                         X_train, y_train,
                                                         X_test, y_test)

# %% Gradient boosting

model_GB = GradientBoostingClassifier(random_state=0)

GB_parameters = {
    'model__n_estimators': [1, 10, 100],
    'model__max_depth': [5, 10, 50, 100],
    'model__subsample': [0.5],
    'model__learning_rate': [0.001, 0.1, 0.5, 0.7],
    }

GB_params, GB_train, GB_validation = train_test_model(model_GB, GB_parameters,
                                                      X_train, y_train,
                                                      X_test, y_test)
