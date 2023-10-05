import random 
import torch
import numpy as np
import os

def set_cuda(use_cuda=False):

    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)



class logs():
    def __init__(self):

        self.noise_type_array     = []
        self.N_T_array            = []
        self.accuracy_array       = []
        self.ewc_lambda_array     = []
        self.lr_array             = []
        self.zeta_array            = []

    def append(self, noise_type, N_T, A_T, lr, ewc_lambda, zeta):
        self.noise_type_array.append(noise_type)
        self.N_T_array.append(N_T)
        self.accuracy_array.append(A_T)
        self.ewc_lambda_array.append(ewc_lambda)
        self.lr_array.append(lr)
        self.zeta_array.append(zeta)

    def write_file(self, file_name):
        # Folder "results" if not already there
        output_folder = "Results_logs"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        file_path = os.path.join(output_folder, file_name)
        with open(file_path, 'w') as log_file: 
            log_file.write('noise_type,N_T,A_T,lr,ewc_lambda,zeta\n')
            for i in range(len(self.noise_type_array)):
                log_file.write('%s, %d, %3.6f,%3.6f,%3.6f,%3.6f\n' %\
                    (self.noise_type_array[i], self.N_T_array[i],self.accuracy_array[i],self.lr_array[i],self.ewc_lambda_array[i],self.zeta_array[i]))
        print('Log file SUCCESSFULLY generated!')



