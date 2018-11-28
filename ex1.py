import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import random
import math
import os
import time

#np.random.seed(100)
#random.seed(100)


def print_img(img_vec, prediction=None, path=None, idx=0):
    img_vec *= 255. # multiply all cells by 255
    img_vec = img_vec.reshape((28, 28))  # reshape to 28x28

    if prediction is not None:
        plt.title('prediction is {prediction}'.format(prediction=prediction))

    if not path:
        plt.imshow(img_vec, cmap='gray')
        plt.show()
    else:
        # save the image
        plt.imsave(path + "/" + str(idx) + ".png", img_vec)
        plt.close()


def svm_train(x, y, lbl_to_class, iterations=100, reg_lambda=0.001, lr=0.0001):
    m = len(x)
    w = np.zeros(x[0].shape).astype(float).reshape(1, -1)  # initialize w to 0

    for iter in range(iterations):
        idx = random.randint(0, m - 1)
        adaptive_lr = lr/math.sqrt(iter+1)
        curr_x = x[idx]
        curr_y = lbl_to_class[y[idx]]
        if 1.0 - (curr_y*np.dot(w[iter,:], curr_x)) >= 0:  # if the classifier is wrong or not confident enough
            new_w = (1-adaptive_lr*reg_lambda)*w[iter,:] + adaptive_lr*curr_y*curr_x
        else:  # if the classifier is right with enough confidence
            new_w = (1-adaptive_lr*reg_lambda)*w[iter,:]
        # save new weight parameters
        w = np.concatenate((w,new_w.reshape(1,-1)), axis=0)

    return np.sum(w, axis=0)


def svm_predict(x, y, lbl_to_class, w):
    num_mistakes = 0
    for curr_x, curr_y in zip(x, y):
        y_hat = np.sign(np.dot(w, curr_x))
        y_true = lbl_to_class[curr_y]
        if y_hat != y_true:
            num_mistakes += 1
    print("percentage of mistakes = " + str(float(num_mistakes)/len(x)))


def precision(preds, labels):

    mistakes = 0
    for pred, label in zip(preds, labels):
        if pred != label:
            mistakes += 1
    return 1 - mistakes/len(labels)


def hamming_decoding(prediction_matrix, M):

    predicitons = []
    for row in prediction_matrix:
        d = np.sum((1 - np.sign(row*M))/2, axis=1)  # compute distance of predictions from each class
        predicitons.append(np.argmin(d))

    return np.asarray(predicitons).astype(float)


def loss_based_decoding(prediction_matrix, M):

    predicitons = []
    for row in prediction_matrix:
        d = np.sum(np.maximum(0, 1 - row*M), axis=1)  # compute distance of predictions from each class
        predicitons.append(np.argmin(d))

    return np.asarray(predicitons).astype(float)


def validation(x_val, y_val, M, params, num_classifiers):

    # Validation: testing all classifiers - build matrix with dims: #examples X #classes
    for curr_class in range(num_classifiers):
        class_conf = np.dot(x_val, params[curr_class])
        prediction_matrix = np.copy(class_conf) if curr_class == 0 else np.concatenate((prediction_matrix, class_conf), axis=1)

    # hamming based decoding
    hamming_predictions = hamming_decoding(prediction_matrix, M)
    hamming_precision = precision(hamming_predictions, y_val)

    # loss based decoding
    loss_based_predictions = loss_based_decoding(prediction_matrix, M)
    loss_based_precision = precision(loss_based_predictions, y_val)

    return hamming_precision, loss_based_precision, prediction_matrix


def testing(x_test, M, params, num_classifiers):
    # Validation: testing all classifiers - build matrix with dims: #examples X #classes
    for curr_class in range(num_classifiers):
        class_conf = np.dot(x_test, params[curr_class])
        prediction_matrix = np.copy(class_conf) if curr_class == 0 else np.concatenate((prediction_matrix, class_conf), axis=1)

    # hamming based decoding
    hamming_predictions = hamming_decoding(prediction_matrix, M)

    # loss based decoding
    loss_based_predictions = loss_based_decoding(prediction_matrix, M)

    return hamming_predictions, loss_based_predictions


def one_vs_all(x, y, x_val, y_val, x_test, path, iterations, reg_lambda, lr):

    # create matrix code
    M = np.asarray([[1.0, -1.0, -1.0, -1.0],
                    [-1.0, 1.0, -1.0, -1.0],
                    [-1.0, -1.0, 1.0, -1.0],
                    [-1.0, -1.0, -1.0, 1.0]])
    params = []

    # Train: training 4 classifiers corresponding to 4 classes
    for curr_class in range(4):
        # map each label to relevant class
        lbl_to_class = {idx: val for idx, val in enumerate(M[curr_class])}
        # train classifier
        w = svm_train(x, y, lbl_to_class, iterations, reg_lambda, lr).reshape(-1, 1)
        params.append(w)

    # validation and testing
    hamming_precision_val, loss_based_precision_val, _ = validation(x_val, y_val, M, params, 4)
    hamming_predictions, loss_based_predictions = testing(x_test, M, params, 4)

    print("one-vs-all - hamming_precision: " + str(hamming_precision_val))
    print("one-vs-all - loss_based_precision: " + str(loss_based_precision_val))

    np.savetxt(path + '/test.onevall.ham.pred' + str('%.4f' % hamming_precision_val), hamming_predictions, delimiter=',')
    np.savetxt(path + '/test.onevall.loss.pred' + str('%.4f' % loss_based_precision_val), loss_based_predictions, delimiter=',')


def all_pairs(x, y, x_val, y_val, x_test, path, iterations, reg_lambda, lr):

    # create matrix code
    M = np.asarray([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                    [0.0, -1.0, 0.0, -1.0, 0.0, 1.0],
                    [0.0, 0.0, -1.0, 0.0, -1.0, -1.0]])
    params = []

    # training 6 classifiers corresponding to all pairs
    for curr_classifier in range(M.shape[1]):
        first_class = float(np.argwhere(M[:, curr_classifier] == 1.0))
        second_class = float(np.argwhere(M[:, curr_classifier] == -1.0))

        curr_x = np.asarray([ex for ex, ey in zip(x, y) if ey == first_class or ey == second_class])
        curr_y = np.asarray([ey for ex, ey in zip(x, y) if ey == first_class or ey == second_class])

        # map each label to relevant class
        lbl_to_class = {first_class: 1.0, second_class: -1.0}
        # train classifier
        w = svm_train(curr_x, curr_y, lbl_to_class, iterations, reg_lambda, lr).reshape(-1, 1)
        params.append(w)

    # validation and testing
    hamming_precision_val, loss_based_precision_val, _ = validation(x_val, y_val, M, params, 6)
    hamming_predictions, loss_based_predictions = testing(x_test, M, params, 6)

    print("all_pairs - hamming_precision: " + str(hamming_precision_val))
    print("all_pairs - loss_based_precision: " + str(loss_based_precision_val))

    np.savetxt(path + '/test.allpairs.ham.pred' + str('%.4f' % hamming_precision_val), hamming_predictions, delimiter=',')
    np.savetxt(path + '/test.allpairs.loss.pred' + str('%.4f' % loss_based_precision_val), loss_based_predictions, delimiter=',')


def random_mat(x, y, x_val, y_val, x_test, path, iterations, reg_lambda, lr, num_classifiers):

    # create matrix code - don't create matrix with all 0's
    M = np.zeros((4, num_classifiers))
    while True:
        M = np.random.randint(-1, 2, (4, num_classifiers))
        if np.all(np.any(M == 1.0, axis=0)) and np.all(np.any(M == -1.0, axis=0)):
            break

    print(M)
    params = []

    # training classifiers
    for curr_classifier in range(M.shape[1]):
        first_class = np.argwhere(M[:, curr_classifier] == 1).flatten().tolist()
        second_class = np.argwhere(M[:, curr_classifier] == -1).flatten().tolist()

        curr_x = np.asarray([ex for ex, ey in zip(x, y) if ey in first_class or ey in second_class])
        curr_y = np.asarray([ey for ex, ey in zip(x, y) if ey in first_class or ey in second_class])

        # map each label to relevant class
        lbl_to_class1 = {c: 1.0 for c in first_class}
        lbl_to_class2 = {c: -1.0 for c in second_class}
        lbl_to_class = {**lbl_to_class1, **lbl_to_class2}

        # train classifier
        w = svm_train(curr_x, curr_y, lbl_to_class, iterations, reg_lambda, lr).reshape(-1, 1)
        params.append(w)

    # validation and testing
    hamming_precision_val, loss_based_precision_val, prediction_matrix = validation(x_val, y_val, M, params, num_classifiers)
    hamming_predictions, loss_based_predictions = testing(x_test, M, params, num_classifiers)

    print("random - hamming_precision: " + str(hamming_precision_val))
    print("random - loss_based_precision: " + str(loss_based_precision_val))

    np.savetxt(path + '/test.randm.ham.pred' + str('%.4f' % hamming_precision_val), hamming_predictions, delimiter=',')
    np.savetxt(path + '/test.randm.loss.pred' + str('%.4f' % loss_based_precision_val), loss_based_predictions, delimiter=',')


if __name__ == '__main__':
    # read data
    mnist = fetch_mldata("MNIST original", data_home="./data")
    eta = 0.1
    X, Y = mnist.data[:60000] / 255., mnist.target[:60000]
    x = [ex for ex, ey in zip(X, Y) if ey in [0, 1, 2, 3]]
    y = [ey for ey in Y if ey in [0, 1, 2, 3]]
    # suffle examples
    x, y = shuffle(x, y, random_state=1)

    # load validation and test sets
    x_val = np.concatenate((np.loadtxt("./code/x_val1.txt"), np.loadtxt("./code/x_val2.txt")))
    y_val = np.concatenate((np.loadtxt("./code/y_val1.txt"), np.loadtxt("./code/y_val2.txt")))

    x_test = np.loadtxt("./code/x4pred.txt")

    exp_index = 1

    # configurations
    iterations = 10000
    reg_lambda = 0.05
    lr = 0.2

    # create directory for writing results
    path = "./predictions/" + str(exp_index)
    os.makedirs(path, exist_ok=True)

    for i in range(15):
        one_vs_all(x, y, x_val, y_val, x_test, path, iterations, reg_lambda, lr)
        all_pairs(x, y, x_val, y_val, x_test, path, iterations, reg_lambda, lr)
        random_mat(x, y, x_val, y_val, x_test, path, iterations, reg_lambda, lr, 15)