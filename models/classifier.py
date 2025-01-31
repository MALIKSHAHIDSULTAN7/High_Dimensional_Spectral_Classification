import torch
import torch.nn as nn

# This was the initial implemention easy to understand but not computationally efficient as it contains for loops.
class Trainable_Difference_Layer(nn.Module):

    def __init__(self, p, k ):
        super(Trainable_Difference_Layer,self).__init__()
        """
        p         -> # of dim / channels /  covariates in a Multi-variate time series data.
        k         -> # of frequencies.
        """
        self.weights = nn.Parameter(torch.ones(k,2*p,2*p))
        self.init_weights()
    def init_weights(self):
        """
        Initialization -> We can either initialize with the inversion
        Because Inverse can be costly we recommend to initialize with zeros
        """
        nn.init.constant_(self.weights.data,0)
    
    def forward(self,z,sigma_1,sigma_2, n_1, n_2):
        """
        Z = (N * freq * 2p)
        sigma_1 = (freq*2p*2p)
        sigma_2 = (freq*2p*2p)
        """
        n_1 = torch.tensor(n_1, dtype = torch.float32)
        n_2 = torch.tensor(n_2, dtype = torch.float32)
        div = n_1 / n_2
        const = torch.log(div)
        sum_freq = self.get_freq_sum(z,sigma_1,sigma_2)
        out = sum_freq.sum(dim = -1)
        out = out + const
        out = out.reshape(-1,1)
        return out

    def get_freq_sum(self,z,sigma_1,sigma_2):

        examples,freq,p = z.shape
        sum_freq = torch.zeros((examples,freq))
        for i in range(examples):
            for j in range(freq):
                z_i = z[i,j,:]
                z_i = z_i.reshape(p, 1)
                sigma_1_i = sigma_1[j,:,:]
                a1 = torch.matmul(z_i.T,self.weights[j,:,:])
                a2 = torch.matmul(a1, z_i)

                a3 = torch.matmul(self.weights[j,:,:],sigma_1_i)
                a4 = a3 + torch.eye(p)

                a5 = torch.det(a4)

                a5 = torch.clamp(a5,min = 1e-5, max = 1e5)

                out = a2 - torch.log(a5)

                sum_freq[i,j] = out
        return sum_freq
    def get_difference_reg(self,sigma_1,sigma_2, lamda):
        """
        lamda -> The regularization parameter controlling Sparsity.
        """
        freq,p,p = sigma_1.shape
        sum_total = torch.zeros(freq)
        regularization      = 0.0
        for i in range(freq):
            sigma_1_i = sigma_1[i,:,:]
            sigma_2_i = sigma_2[i,:,:]

            sigma_1_Dk = torch.matmul(sigma_1_i,self.weights[i,:,:])
            Dk_sigma_2 = torch.matmul(self.weights[i,:,:],sigma_2_i)

            trace_1    = torch.trace(torch.matmul(sigma_1_Dk,Dk_sigma_2.T))

            sigma_2_Dk = torch.matmul(sigma_2_i,self.weights[i,:,:])
            Dk_sigma_1 = torch.matmul(self.weights[i,:,:],sigma_1_i)

            trace_2    = torch.trace(torch.matmul(sigma_2_Dk,Dk_sigma_1.T))

            trace      = 1/4 * (trace_1 + trace_2)

            sigma_1_sigma_2 = sigma_1_i - sigma_2_i
            Dk_sigma12 = torch.trace(torch.matmul(self.weights[i,:,:],sigma_1_sigma_2.T))

            theta      = trace - Dk_sigma12
            sum_total[i] = theta

            regularization += torch.abs(self.weights[i,:,:]).sum()
        sum_total_sum = sum_total.sum()
        reg           = lamda * regularization
        diff_loss     = reg + sum_total_sum
        return diff_loss, sum_total_sum
    

# This is a vectorized implementation of the same classifier and is quick as it avoids the un-necessary for loops.


class Trainable_Difference_Layer_Vectorized(nn.Module):
    def __init__(self, p, k):
        super(Trainable_Difference_Layer_Vectorized, self).__init__()
        """
        p -> # of dimensions (2p)
        k -> # of frequencies
        """
        # Initialize weights as nn.Parameter, which are learnable parameters
        self.weights = nn.ParameterList([nn.Parameter(torch.ones(2 * p, 2 * p)) for _ in range(k)])
        self.init_weights()

    def init_weights(self):
        """
        Initialization -> We initialize the weights to 0.0 initially.
        """
        for weight in self.weights:
            nn.init.constant_(weight.data, 0.0)

    def forward(self, z, sigma_1, sigma_2, n_1, n_2):
        """
        Forward pass for the layer using separate weight matrices for each frequency.
        
        z = (examples, freq, 2p)       
        sigma_1 = (freq, 2p, 2p)         
        sigma_2 = (freq, 2p, 2p)         
        n_1, n_2 = scalar inputs describing  the # of training samples for both classes.
        """
        n_1 = torch.tensor(n_1, dtype=torch.float32)
        n_2 = torch.tensor(n_2, dtype=torch.float32)
        div = n_1 / n_2
        const = torch.log(div)
        sum_freq = self.get_freq_sum(z, sigma_1, sigma_2) 
        out = sum_freq.sum(dim=-1) + const
        out = out.reshape(-1, 1)
        return out

    def get_freq_sum(self, z, sigma_1, sigma_2):
        """
        Z = (examples * freq * 2p)
        sigma_1 = (freq*2p*2p)
        sigma_2 = (freq*2p*2p)
        """
        examples, freq, p = z.shape
        sum_freq = torch.zeros((examples, freq)).to(z.device)

        # Iterate over the frequencies 
        for k in range(freq):
            
            weight_k = self.weights[k]  # Shape (2p, 2p)

            
            z_k = z[:, k, :]  # Shape (examples, 2p)
            sigma_1_k = sigma_1[k, :, :]  # Shape (2p, 2p)
            sigma_2_k = sigma_2[k, :, :]  # Shape (2p, 2p)

           
            z_k = z_k.unsqueeze(2)  # Shape (examples, 2p, 1)
            a1 = torch.matmul(z_k.transpose(1, 2), weight_k)  # (examples, 1, 2p)
            a2 = torch.matmul(a1, z_k)  # (examples, 1, 1)
            a3 = torch.matmul(weight_k, sigma_1_k)  # (2p, 2p)
            a4 = a3 + torch.eye(p).to(z.device)  # Adding identity matrix (2p, 2p)
            a5 = torch.det(a4)  # Compute determinant

            a5 = torch.clamp(a5, min=1e-5, max=1e5)  
            sum_freq[:, k] = a2.squeeze() - torch.log(a5)  

        return sum_freq

    def get_difference_reg(self, sigma_1, sigma_2, lamda):
        """
        Compute the d trace loss
        """
        freq, p, _ = sigma_1.shape
        sum_total = torch.zeros(freq).to(device=sigma_1.device)
        regularization = 0.0

        # Iterate over each frequency to calculate the regularization term
        for i in range(freq):
            sigma_1_i = sigma_1[i, :, :]
            sigma_2_i = sigma_2[i, :, :]

            sigma_1_Dk = torch.matmul(sigma_1_i, self.weights[i])
            Dk_sigma_2 = torch.matmul(self.weights[i], sigma_2_i)

            trace_1 = torch.trace(torch.matmul(sigma_1_Dk, Dk_sigma_2.T))

            sigma_2_Dk = torch.matmul(sigma_2_i, self.weights[i])
            Dk_sigma_1 = torch.matmul(self.weights[i], sigma_1_i)

            trace_2 = torch.trace(torch.matmul(sigma_2_Dk, Dk_sigma_1.T))

            trace = 1 / 4 * (trace_1 + trace_2)

            sigma_1_sigma_2 = sigma_1_i - sigma_2_i
            Dk_sigma12 = torch.trace(torch.matmul(self.weights[i], sigma_1_sigma_2.T))

            theta = trace - Dk_sigma12
            sum_total[i] = theta

            regularization += torch.abs(self.weights[i]).sum()

        sum_total_sum = sum_total.sum()
        reg = lamda * regularization
        diff_loss = reg + sum_total_sum
        return diff_loss, sum_total_sum
