import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score


def train(model,criterion,optimizer,data_loader,lamda,sigma_1,sigma_2,n_1, n_2,only_reg = True):
    """
    model -> The model to be trained
    criterion -> BCE Loss will be used if only_reg = False, this is not recommended.
    optimizer -> The Gradient Based Optimizer used to train the model ADAM with default parameters is recommended
    data_loader -> The data loader to load the x,y training pairs
    lamda -> The regularization hyper-parameter for L1 Regularization
    only_reg -> If set to true only D-Trace Loss will be used

    """
    model.train()
    loss = 0
    accuracy = 0
    for x,y in data_loader:

        if only_reg == True:
            ls , dt_loss       = model.get_difference_reg(sigma_1,sigma_2,lamda)
        elif only_reg == False:
            y_probs = model(x,sigma_1,sigma_2,n_1,n_2)
            y_probs   = y_probs.reshape(-1,1)
            y_probs   = nn.functional.sigmoid(y_probs)
            y         = y.reshape(-1,1)
            ls        = criterion(y_probs,y)
            l,dt_loss  = model.get_difference_reg(sigma_1,sigma_2,lamda)
            ls += l
        optimizer.zero_grad()
        ls.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
        optimizer.step()
        loss += dt_loss.item()

    loss /= len(data_loader)
    accuracy /= len(data_loader)
    return loss,accuracy




def evaluate(model,criterion,data_loader,lamda,sigma_1,sigma_2,n_1,n_2):
    model.eval()
    accuracy = 0
    with torch.no_grad():
        loss, dt_loss        = model.get_difference_reg(sigma_1,sigma_2,lamda)
    return dt_loss.item(),accuracy






def tune_lambda_and_evaluate(X, y, lambdas):
    loo = LeaveOneOut()
    best_lambda = None
    best_accuracy = 0

    for lam in lambdas:
        accuracies = []

        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Apply lambda transformation
            X_train_transformed = (X_train > lam).astype(int)
            X_test_transformed = (X_test > lam).astype(int)
            accuracies.append(accuracy_score(y_test, X_test_transformed))

        mean_accuracy = np.mean(accuracies)

        # Check if the current lambda is the best
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_lambda = lam

    return best_lambda, best_accuracy