# %% importing

from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import cv2
from skimage.transform import resize
import matplotlib.pyplot as plt
import pandas as pd


# %% functions

# The new size(width or height is the most common value in the histogram
# lst - list of sizes (width of heights) of the images
def hist_wh(lst):
    hist_wh = np.histogram(lst)
    new_idx = np.where(hist_wh[0] == np.max(hist_wh[0]))
    new_size = np.round((hist_wh[1][new_idx[0][0]] + hist_wh[1][new_idx[0][0] + 1]) / 2)
    return new_size


# assigning all images of a certain class into a numpy array
# pattern - pattern of class name, w - width of images, h - height of images, names - of images, path - path of images
def class2mat(pattern, w, h, names, path):
    class_images = list(filter(lambda x: pattern in x, names))
    full_mat = np.zeros((int(h), int(w), 75))

    for j in range(0, len(class_images)):
        class_img_path = path + '/' + class_images[j]
        class_img = cv2.imread(class_img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = resize(class_img, (h, w))
        full_mat[:, :, j] = img_resized
    return full_mat


# fitting the model classifier using GridSearchCV + testing the model and printing results
def train_test_model(model, parameters, train_set, train_labels, test_set, test_labels):
    # scaling
    general_scaler = StandardScaler()
    train_set = general_scaler.fit_transform(train_set)
    test_set = general_scaler.transform(test_set)

    # training
    pipe = Pipeline(steps=[("model", model)])
    scoring = {"AUC": "roc_auc"}
    clf = GridSearchCV(pipe, parameters, scoring=scoring,
                       refit="AUC", return_train_score=True, n_jobs=-1, cv=5)
    clf.fit(train_set, train_labels)

    # results of hyper parameter tuning
    best_params = clf.best_params_

    tuning_results = pd.DataFrame(clf.cv_results_)

    maximal_idx = tuning_results['mean_test_AUC'].idxmax()

    train_results = tuning_results[['split0_train_AUC', 'split1_train_AUC', 'split2_train_AUC',
                                    'split3_train_AUC', 'split4_train_AUC',
                                    'mean_train_AUC', 'std_train_AUC']]
    validation_results = tuning_results[['split0_test_AUC', 'split1_test_AUC', 'split2_test_AUC',
                                         'split3_test_AUC', 'split4_test_AUC',
                                         'mean_test_AUC', 'std_test_AUC']]
    best_train_results = pd.DataFrame(train_results.iloc[maximal_idx])
    best_validation_results = pd.DataFrame(validation_results.iloc[maximal_idx])

    # testing
    y_preds = clf.predict(test_set)

    # roc curve
    fpr, tpr, thresholds = metrics.roc_curve(test_labels, y_preds)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    display.plot()
    plt.title('ROC curve')
    plt.show()

    # confusion matrix
    cm = confusion_matrix(test_labels, y_preds, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=clf.classes_)
    cmap = 'Greens'
    disp.plot(cmap=cmap)
    plt.show()

    return best_params, best_train_results, best_validation_results
