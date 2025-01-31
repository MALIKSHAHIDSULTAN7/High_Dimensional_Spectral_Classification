import numpy as np
import h5py

def get_high_dimensional_data(path, dataset_name, return_raw_data = False):
    f = h5py.File(path, 'r')

    dataset = f[dataset_name]
    keys = ['XRawTS', 'YRawTS', 'hat_tildeS1', 'hat_tildeS2', 'tildeXDFT', 'tildeYDFT']

    del f

    if return_raw_data:
        class_1_data = dataset['XRawTS']
        class_2_data = dataset['YRawTS']
        return class_1_data, class_2_data



    sigma_1 = dataset['hat_tildeS1']
    sigma_2 = dataset['hat_tildeS2']

    z_class_1 = dataset['tildeXDFT']

    z_class_1_train = []
    z_class_1_test  = []
    examples = z_class_1.shape[-1]//2
    for i in range(z_class_1.shape[-1]):
        if i < examples:
            z_class_1_train.append(z_class_1[:,:,i])
        elif i >= examples:
            z_class_1_test.append(z_class_1[:,:,i])

    z_class_2 = dataset['tildeYDFT']
    z_class_2_train = []
    z_class_2_test  = []

    examples = z_class_2.shape[-1]//2
    for i in range(z_class_2.shape[-1]):
        if i < examples:
            z_class_2_train.append(z_class_2[:,:,i])
        elif i >= examples:
            z_class_2_test.append(z_class_2[:,:,i])


    return sigma_1, sigma_2, np.array(z_class_1_train), np.array(z_class_2_train), np.array(z_class_1_test) , np.array(z_class_2_test)


def stack_data(class_1, class_2):
    combined = np.vstack([class_1, class_2])
    labels   = np.zeros((combined.shape[0],1))
    labels[:class_1.shape[0]] = 1.0
    return combined, labels

def get_differences(sigma_1,sigma_2,T):
    # function to rank frequencies
    freq,_,_ = sigma_1.shape
    norms    = []
    for i in range(freq):
        s1 = sigma_1[i,:,:]
        s2 = sigma_2[i,:,:]
        s1 = s1 / (2*(1 - np.cos(2*np.pi*(freq / T))))
        vec = s1 - s2
        norms.append(np.linalg.norm(vec))
    norms = np.array(norms).reshape(-1,1)
    indices = np.argsort(norms, axis = 0).reshape(-1)
    indices = list(indices)
    ordered_norms = norms[indices,:]
    ks = []
    for i in range(freq):
        if i+1 < freq-1:
            r = ordered_norms[i+1,:]/ ordered_norms[i,:]
            ks.append(r)
    ks = np.array(ks).reshape(-1,1)
    ks = ks[1:,:]
    opt_k = np.argmax(ks, axis = 0)
    return ks, opt_k, indices[opt_k.item():], indices

