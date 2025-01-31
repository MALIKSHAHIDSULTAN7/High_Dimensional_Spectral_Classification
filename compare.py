import numpy as np
import torch
import torch.nn as nn 
from utils.training_utils import tune_lambda_and_evaluate
import numpy as np
import random
import torch
import os
from models.classifier import Trainable_Difference_Layer, Trainable_Difference_Layer_Vectorized
from utils.data_utils import get_high_dimensional_data, get_differences, stack_data
from utils.training_utils import train, evaluate
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
torch.manual_seed(123)
import warnings
torch.use_deterministic_algorithms(True)
random.seed(123)
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score
import seaborn as sns



"""path = '/Users/sultm0a/Documents/NeurIPS_High_Dim_EX1A/Ex1A.hd5'
Data_Name = 'EX1A'
only_reg = True

data_sets_indexes = ['ITER_1']
np.random.seed(123)
torch.manual_seed(123)
epochs = 40
save_path = '/Users/sultm0a/High_Dimensional_Spectral_Classification/Results'
save_path_data = '/Users/sultm0a/Documents/NeurIPS_High_Dim_'+ str(Data_Name) + '/Data/'
for dataset_name in data_sets_indexes:
    print('Working on Data Set {}'.format(dataset_name))
    print(60*'#')
    sigma_1, sigma_2, z_class_1_train,z_class_2_train, z_class_1_test , z_class_2_test = get_high_dimensional_data(path = path,
                                                                                                                   dataset_name = dataset_name , return_raw_data = False)

    sigma_1 = np.array(sigma_1)
    sigma_2 = np.array(sigma_2)

    print('Sigmas')
    print(sigma_1.shape)
    print(sigma_2.shape)
    device = 'cpu'


    p = int(sigma_1.shape[-1]//2)
    freq = sigma_1.shape[0]
    print('Dim {}'.format(p))
    print('Freq {}'.format(freq))
    sigma_1_data = torch.tensor(np.array(sigma_1),dtype = torch.float32, device = device)
    sigma_2_data = torch.tensor(np.array(sigma_2),dtype = torch.float32, device = device)
    x_train,y_train = stack_data(z_class_1_train,z_class_2_train)
    x_test,y_test   = stack_data(z_class_1_test,z_class_2_test)
    x_val , x_test, y_val, y_test = train_test_split(x_test,y_test, test_size = 0.5, random_state=123, stratify=y_test)
    n_1 = y_train.sum()
    n_2 = y_train.shape[0] - n_1
    print('Train')
    print(x_train.shape)
    print(y_train.shape)
    print('N1 ', n_1)
    print('N2 ', n_2)

    mu = np.mean(x_train, keepdims= True, axis = 0)
    std = np.std(x_train, keepdims= True, axis = 0)
    x_train = (x_train - mu) / (std + 1e-4)
    x_test  = (x_test - mu) / (std + 1e-4)
    x_val   = (x_val - mu) / (std + 1e-4)

    x_train_path = 'X_train_' + Data_Name + '_' + dataset_name + '_.npy'
    y_train_path = 'Y_train_' + Data_Name + '_' + dataset_name + '_.npy'

    x_test_path = 'X_test_' + Data_Name + '_' + dataset_name + '_.npy'
    y_test_path = 'Y_test_' + Data_Name + '_' + dataset_name + '_.npy'

    x_val_path = 'X_val_' + Data_Name + '_' + dataset_name + '_.npy'
    y_val_path = 'Y_val_' + Data_Name + '_' + dataset_name + '_.npy'

    x_train_save_loc = os.path.join(save_path_data,x_train_path)
    y_train_save_loc = os.path.join(save_path_data,y_train_path)

    x_test_save_loc = os.path.join(save_path_data,x_test_path)
    y_test_save_loc = os.path.join(save_path_data,y_test_path)

    x_val_save_loc = os.path.join(save_path_data,x_val_path)
    y_val_save_loc = os.path.join(save_path_data,y_val_path)




    X_train = torch.tensor(x_train, dtype= torch.float32, device = device)
    Y_train = torch.tensor(y_train, dtype= torch.float32, device = device)
    print(X_train.shape)
    print(Y_train.shape)
    train_dataset = TensorDataset(X_train,Y_train)


    X_val = torch.tensor(x_val, dtype= torch.float32, device = device)
    Y_val = torch.tensor(y_val, dtype= torch.float32, device = device)
    val_dataset = TensorDataset(X_val,Y_val)

    batch_size = 32
    train_data_loader = DataLoader(train_dataset,batch_size = batch_size, shuffle = True)
    val_data_loader = DataLoader(val_dataset, batch_size= x_val.shape[0], shuffle = False)



Lz_non_vec = model_non_vec(X_train,sigma_1_data,sigma_2_data,n_1,n_2)
Lz_vec = model_vec(X_train,sigma_1_data,sigma_2_data,n_1,n_2)

lambdas_non_vec = np.linspace(max(Lz_non_vec).detach().numpy(), min(Lz_non_vec).detach().numpy(), 100)
lambdas_vec = np.linspace(max(Lz_vec).detach().numpy(), min(Lz_vec).detach().numpy(), 100)

best_lambda_non_vec, best_accuracy_non_vec = tune_lambda_and_evaluate(Lz_non_vec.detach().numpy(), Y_train.detach().numpy(), lambdas_non_vec)
print(f"Best Lambda Non Vec: {best_lambda_non_vec}")
print(f"Best Accuracy Non Vec: {best_accuracy_non_vec}")

best_lambda_vec, best_accuracy_vec = tune_lambda_and_evaluate(Lz_vec.detach().numpy(), Y_train.detach().numpy(), lambdas_vec)
print(f"Best Lambda  Vec: {best_lambda_vec}")
print(f"Best Accuracy Vec: {best_accuracy_vec}")

Lz_test_non_vec = model_non_vec(torch.tensor(x_test, dtype = torch.float32),sigma_1_data,sigma_2_data, n_1, n_2)
Lz_test_vec = model_vec(torch.tensor(x_test, dtype = torch.float32),sigma_1_data,sigma_2_data, n_1, n_2)

preds_non_vec = (Lz_test_non_vec.detach().numpy() > best_lambda_non_vec).astype(int)
preds_vec = (Lz_test_vec.detach().numpy() > best_lambda_vec).astype(int)
print('Test Set Non Vec Acc {}'.format(accuracy_score(y_test, preds_non_vec)))

print('Test Set Vec Acc {}'.format(accuracy_score(y_test, preds_vec)))

"""
model_non_vec = torch.load('/Users/sultm0a/High_Dimensional_Spectral_Classification/Results/EX1A_Non_Vec_Dtrace_ITER_1_0.001.pth')
model_vec     = torch.load('/Users/sultm0a/High_Dimensional_Spectral_Classification/Results/EX1A_Vec_Dtrace_ITER_1_0.001.pth')

weights_non_vec = model_non_vec.weights.data.detach().numpy()
weights_vec     = np.array([weight.data.detach().numpy() for weight in model_vec.weights])
norms_vec = []
norms_non_vec = []
for i in range(125):
  norms_vec.append(np.linalg.norm(weights_vec[i,:,:]))
  norms_non_vec.append(np.linalg.norm(weights_non_vec[i,:,:]))
plt.plot(norms_non_vec, label = 'Non Vec')
plt.plot(norms_vec , label  = 'Vec')
plt.show()

for i in range(0,125):
  fig, ax = plt.subplots(1,2)
  sns.heatmap(weights_non_vec[i,:,:], ax = ax[0])
  sns.heatmap(weights_non_vec[i,:,:], ax = ax[1])
  plt.show()