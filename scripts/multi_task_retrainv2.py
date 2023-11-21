import os
import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np
import datetime
import copy
import torch.optim.lr_scheduler as lr_scheduler

import torch
from torch import nn, optim
from torchvision import datasets, transforms

from utils.common import set_random_seeds, set_cuda, logs
from utils.dataloaders import pytorch_dataloader, pytorch_rotation_dataloader
from utils.dataloaders import cifar_c_dataloader, imagenet_c_dataloader 
from utils.dataloaders import auxilary_samples_dataloader_iterative, auxilary_samples_dataloader_iterative_imagenet

from utils.multi_task_model import model_selection

from utils.model import train_model

from metrics.accuracy.topAccuracy import top1Accuracy, top1Accuracy_rotation

from methods.EWC_multi import on_task_update, train_model_ewc

import matplotlib.pyplot as plt
from math import log

import pickle

# =====================================================
# == Declarations
# =====================================================
SEED_NUMBER              = 0
USE_CUDA                 = True


DATASET_DIR              = '../../NetZIP/datasets/ImageNet/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC'#'../datasets/CIFAR10/'#'../../NetZIP/datasets/TinyImageNet/' #'../datasets/CIFAR100/' # '../../NetZIP/datasets/ImageNet/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC' 
DATASET_NAME             = "ImageNet" # Options: "CIFAR10" "CIFAR100" "TinyImageNet"  "ImageNet"

NUM_CLASSES              = 1000 # Number of classes in dataset

MODEL_CHOICE             = "resnet" # Option:"resnet" "vgg"
MODEL_VARIANT            = "resnet18" # Common Options: "resnet18" "vgg11" For more options explore files in models to find the different options.

MODEL_DIR                = "../models/" + MODEL_CHOICE
MODEL_SELECTION_FLAG     = 2 # create an untrained model = 0, start from a pytorch trained model = 1, start from a previously saved local model = 2

MODEL_FILENAME     = MODEL_VARIANT +"_"+DATASET_NAME+".pt"
MODEL_FILEPATH     = os.path.join(MODEL_DIR, MODEL_FILENAME)


# NOISE_TYPES_ARRAY = ["brightness","contrast","defocus_blur",
# 					"elastic_transform","fog","frost","gaussian_blur",
# 					"gaussian_noise", "glass_blur", "impulse_noise",
# 					"jpeg_compression", "motion_blur", "pixelate", 
# 					"saturate", "shot_noise", "snow", "spatter", 
# 					"speckle_noise", "zoom_blur"]

NOISE_TYPES_ARRAY = ["brightness","contrast","defocus_blur",
					"elastic_transform","fog","frost",
					"gaussian_noise", "glass_blur", "impulse_noise",
					"jpeg_compression", "motion_blur", "pixelate", 
					"shot_noise", "snow", "zoom_blur"]

# NOISE_TYPES_ARRAY = ["gaussian_noise"]#,"shot_noise"]

# NOISE_TYPES_ARRAY = ["jpeg_compression", "motion_blur", "pixelate", "shot_noise", "snow", "zoom_blur"]
# NOISE_TYPES_ARRAY = ["contrast","motion_blur","fog"]

# NOISE_TYPES_ARRAY = ["impulse_noise"]

NOISE_SEVERITY 	  = 5 # Options from 1 to 5

MAX_SAMPLES_NUMBER = 120 #220
N_T_Initial        = 1
N_T_STEP = 25 #16

def retrain(model, testloader, N_T_trainloader_c, N_T_testloader_c, device, fisher_dict, optpar_dict, num_retrain_epochs, lambda_retrain, lr_retrain, zeta, layers_keywords, fix_batch_noramlisation):
	
	model.use_rotation_head()
	# model.use_classification_head()

	# Copy model for retraining
	retrained_model = copy.deepcopy(model)

	# retrained_model.use_classification_head()
	retrained_model.use_rotation_head()
	
	# Retrain
	retrained_model = train_model_ewc(model = retrained_model, 
									layers_keywords = layers_keywords,
									train_loader = N_T_trainloader_c, 
									test_loader = N_T_testloader_c, 
									device = device, 
									fisher_dict = fisher_dict,
									optpar_dict = optpar_dict,
									num_epochs=num_retrain_epochs, 
									ewc_lambda = lambda_retrain,
									learning_rate=lr_retrain, 
									momentum=0.9, 
									weight_decay=1e-5,
									fix_batch_noramlisation=fix_batch_noramlisation)


	# ========================================	
	# == Evaluate Retrained model
	# ========================================
	retrained_model.use_rotation_head()
	# Calculate accuracy of retrained model on target domain samples for the task of rotation 
	_, A_k    = top1Accuracy(model=retrained_model, test_loader=N_T_testloader_c, device=device, criterion=None)
	print("A_k = ",A_k)

	retrained_model.use_classification_head()
	# Calculate accuracy of retrained model on initial test dataset for the task of classification  
	_, A_0    = top1Accuracy(model=retrained_model, test_loader=testloader, device=device, criterion=None)
	print("A_0 = ",A_0)

	if isinstance(A_0, torch.Tensor):
		A_0 = A_0.cpu().numpy()

	if isinstance(A_k, torch.Tensor):
		A_k = A_k.cpu().numpy()

	# Calculate CFAS
	# CFAS = A_k.cpu().numpy() * (zeta*A_0.cpu().numpy() +1)
	CFAS = A_k + zeta*A_0
	# CFAS = A_k.cpu().numpy() + 2*A_0.cpu().numpy()
	# CFAS = A_k.cpu().numpy() + 1.5*A_0.cpu().numpy()
	# CFAS = A_k.cpu().numpy() * (5*A_0.cpu().numpy() +1)
	# CFAS = A_k.cpu().numpy() * (10*A_0.cpu().numpy() +1)

	return retrained_model, CFAS

def main():
	# ========================================
	# == Experiment Settings.
	# ========================================
	num_retrain_epochs = 10

	zeta = 10

	experiment_number = 26
	fix_batch_noramlisation = True
	Swap_layers_starting_from = 5 # 2 : From layers 2  |  3 : From layers 3  |  4 : From layers 4  |  5 : For output layers only

	if Swap_layers_starting_from == 5:

		layers_keywords = ["avgpool",               "fc", 
	                       "classification_avgpool","classification_head", 
	                       "rotation_avgpool",      "rotation_head"] # Use For swapping output layers only

	if Swap_layers_starting_from == 4:
		layers_keywords = ["layer4",                "avgpool",               "fc", 
	                       "classification_layer4", "classification_avgpool","classification_head", 
	                       "rotation_layer4",       "rotation_avgpool",      "rotation_head"] # Use For swapping from layers 4 to end
	
	if Swap_layers_starting_from == 3:                       
		layers_keywords = ["layer3",               "layer4",                "avgpool",               "fc", 
	                       "classification_layer3","classification_layer4", "classification_avgpool","classification_head", 
	                       "rotation_layer3",      "rotation_layer4",       "rotation_avgpool",      "rotation_head"]  # Use For swapping from layers 3 to end

	if Swap_layers_starting_from == 2:
		layers_keywords = ["layer2" ,              "layer3",               "layer4",                "avgpool",               "fc", 
	                       "classification_layer2","classification_layer3","classification_layer4", "classification_avgpool","classification_head", 
	                       "rotation_layer2",      "rotation_layer3",      "rotation_layer4",       "rotation_avgpool",      "rotation_head"]  # Use For swapping from layers 2 to end

	# ========================================
	# == Preliminaries
	# ========================================
	# Fix seeds to allow for repeatable results 
	set_random_seeds(SEED_NUMBER)

	# Setup device used for training either gpu or cpu
	device = set_cuda(USE_CUDA)

	# Load model
	model = model_selection(model_selection_flag=MODEL_SELECTION_FLAG, model_dir=MODEL_DIR, model_choice=MODEL_CHOICE, model_variant=MODEL_VARIANT, saved_model_filepath=MODEL_FILEPATH, num_classes=NUM_CLASSES, device=device, mutli_selection_flag = False)
	model_multi = model_selection(model_selection_flag=MODEL_SELECTION_FLAG, model_dir=MODEL_DIR, model_choice=MODEL_CHOICE, model_variant=MODEL_VARIANT, saved_model_filepath=MODEL_FILEPATH, num_classes=NUM_CLASSES, device=device, mutli_selection_flag = True, Swap_layers_starting_from = Swap_layers_starting_from)
	# model_multi_copy = copy.deepcopy(model_multi)
	print("Progress: Model has been setup.")

	# Setup original dataset
	if DATASET_NAME == "ImageNet":
		retraining_imagenet_flag = True
	else:
		retraining_imagenet_flag = False

	trainloader_clas, testloader_clas, small_testloader_clas = pytorch_dataloader(dataset_name=DATASET_NAME, dataset_dir=DATASET_DIR, batch_size=64, retraining_imagenet=retraining_imagenet_flag)
	trainloader_rot, testloader_rot = pytorch_rotation_dataloader(dataset_name=DATASET_NAME, dataset_dir=DATASET_DIR, batch_size=64) 
	# trainloader_withRot, testloader_withRot = pytorch_dataloader_with_rotation(dataset_name=DATASET_NAME, dataset_dir=DATASET_DIR, images_size=32, batch_size=64, do_rotations=True)
	print("Progress: Dataset Loaded.")

	# accuracies = []
	print(model_multi)
	# Evaluate model
	_, eval_accuracy_clas_single   = top1Accuracy(model=model, test_loader=testloader_clas, device=device, criterion=None)
	_, eval_accuracy_clas_multi   = top1Accuracy(model=model_multi, test_loader=testloader_clas, device=device, criterion=None)
	# _, eval_accuracy_clas_multi_copy   = top1Accuracy(model=model_multi_copy, test_loader=testloader_clas, device=device, criterion=None)
	model_multi.use_rotation_head()
	_, eval_accuracy_rot_multi  = top1Accuracy(model=model_multi, test_loader=testloader_rot, device=device, criterion=None)
	
	print("Single-Head Model Classificaiton Accuray on original dataset = ",eval_accuracy_clas_single)
	print("Multiple-Head Model Classificaiton Accuray on original dataset = ",eval_accuracy_clas_multi)
	print("Multiple-Head Model Rotation Accuray on original dataset = ",eval_accuracy_rot_multi)


	# Train the Rotation Head: Freeze all layers except the layers for the roation auxilary task.
	for param in model_multi.parameters():
		param.requires_grad = False

	for param in model_multi.rotation_head.parameters():#resnet.fc.parameters():
		param.requires_grad = True

	# Train new layers in model for rotation task
	model_multi = train_model(model=model_multi, train_loader=trainloader_rot, test_loader=testloader_rot, device=device, learning_rate=1e-2, num_epochs=2, fix_batch_noramlisation=True)

	# Once new layers are trained set grad back to usual
	for param in model_multi.parameters():#resnet.fc.parameters():
		param.requires_grad = True

	# Evaluate model
	model_multi.use_classification_head()
	_, eval_accuracy_clas_single   = top1Accuracy(model=model, test_loader=testloader_clas, device=device, criterion=None)
	_, eval_accuracy_clas_multi   = top1Accuracy(model=model_multi, test_loader=testloader_clas, device=device, criterion=None)
	model_multi.use_rotation_head()
	_, eval_accuracy_rot_multi  = top1Accuracy(model=model_multi, test_loader=testloader_rot, device=device, criterion=None)
	
	print("Single-Head Model Classificaiton Accuray on original dataset = ",eval_accuracy_clas_single)
	print("Multiple-Head Model Classificaiton Accuray on original dataset = ",eval_accuracy_clas_multi)
	print("Multiple-Head Model Rotation Accuray on original dataset = ",eval_accuracy_rot_multi)

	# Set model to classificaton head
	model_multi.use_classification_head()

	# Initiate dictionaries for regularisation using EWC	
	fisher_dict = {}
	optpar_dict = {}

	optimizer = optim.SGD(model_multi.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
	
	save_dict = True
	load_dict = False

	if load_dict == True and (DATASET_NAME == "ImageNet" or DATASET_NAME == "TinyImageNet"):
		with open("resnet18_"+DATASET_NAME+"_layers_frozen_from_"+str(Swap_layers_starting_from)+"_fisher.pkl", 'rb') as file:
			fisher_dict = pickle.load(file)

		with open("resnet18_"+DATASET_NAME+"_layers_frozen_from_"+str(Swap_layers_starting_from)+"_optpar.pkl", 'rb') as file:
			optpar_dict = pickle.load(file)

		print("PROGRESS: Calculated Fisher loaded")

	else:
		# fisher_dict, optpar_dict = on_task_update(0, trainloader_clas, model, optimizer, fisher_dict, optpar_dict, device)
		fisher_dict, optpar_dict = on_task_update(0, trainloader_clas, model_multi, optimizer, fisher_dict, optpar_dict, device, layers_keywords=layers_keywords)
		print("PROGRESS: Calculated Fisher matrix")

		if save_dict == True and (DATASET_NAME == "ImageNet" or DATASET_NAME == "TinyImageNet"):

			with open("resnet18_"+DATASET_NAME+"_layers_frozen_from_"+str(Swap_layers_starting_from)+"_fisher.pkl", 'wb') as file:
				pickle.dump(fisher_dict, file)

			with open("resnet18_"+DATASET_NAME+"_layers_frozen_from_"+str(Swap_layers_starting_from)+"_optpar.pkl", 'wb') as file:
				pickle.dump(optpar_dict, file)

			print("PROGRESS: Saved Fisher matrix")



	# ========================================
	# == Load Noisy Data
	# ========================================
	results_log = logs() 

	for noise_type in NOISE_TYPES_ARRAY:
		if noise_type == "original":
			testloader_c = testloader
		else:
			# load noisy dataset
			if DATASET_NAME == "CIFAR10" or DATASET_NAME == "CIFAR100":
				trainloader_c_clas, testloader_c_clas, noisy_images, noisy_labels      = cifar_c_dataloader(NOISE_SEVERITY, noise_type, DATASET_NAME)

			elif DATASET_NAME == "ImageNet":
				trainloader_c_clas, testloader_c_clas, trainset_c_clas, testset_c_clas = imagenet_c_dataloader(NOISE_SEVERITY, noise_type, tiny_imagenet=False)

			elif DATASET_NAME == "TinyImageNet":
				trainloader_c_clas, testloader_c_clas, trainset_c_clas, testset_c_clas = imagenet_c_dataloader(NOISE_SEVERITY, noise_type, tiny_imagenet=True)
			# print("shape of noisy images = ", np.shape(noisy_images))
			# print("shape of noisy labels = ", np.shape(noisy_labels))
		
		# Evaluate testloader_c on original model
		model_multi.use_classification_head()
		_, initial_A_T    = top1Accuracy(model=model_multi, test_loader=testloader_c_clas, device=device, criterion=None)
		print("Cls Accuray on " + noise_type +" dataset = ",initial_A_T)

		# ========================================	
		# == Append Details of model performance before retraining
		# ========================================
		results_log.append(noise_type, 0, initial_A_T,
								0, 
								0, 
								0)
		# ========================================	
		# == Select random N_T number of datapoints from noisy data for retraining
		# ========================================
		# N_T_vs_A_T = []
		samples_indices_array = []
		# Extract N_T random samples
		for N_T in range(0, MAX_SAMPLES_NUMBER, N_T_STEP):
			if N_T == 0:
				N_T = N_T_Initial

			print("++++++++++++++")
			print("N_T = ", N_T)
			print("Noise Type = ", noise_type) 
			if DATASET_NAME == "CIFAR10" or DATASET_NAME == "CIFAR100": 
				N_T_trainloader_c_rot, N_T_testloader_c_rot, samples_indices_array = auxilary_samples_dataloader_iterative(N_T, noisy_images, samples_indices_array)
			elif DATASET_NAME == "ImageNet" or DATASET_NAME == "TinyImageNet":
				N_T_trainloader_c_rot, N_T_testloader_c_rot, samples_indices_array = auxilary_samples_dataloader_iterative_imagenet(N_T, testset_c_clas, samples_indices_array)
			# TODO: the commented two lines above need to get sorted to load data for ImageNet


			# ========================================	
			# == Retrain model
			# ========================================
			temp_list_retrained_models = []
			temp_list_lr               = []
			temp_list_lambda           = []
			temp_list_CFAS             = []

			# SGC_flag  = False
			# CFAS_flag = False
			EWC_flag  = True

			# if SGC_flag == True:
			# 	lr_retrain = 1e-2
			# 	lambda_retrain = 0

			# 	retrained_model, CFAS = retrain(model, testloader, N_T_trainloader_c, N_T_testloader_c, device, fisher_dict, optpar_dict, num_retrain_epochs, lambda_retrain, lr_retrain, zeta)
				
			# 	# Append Data
			# 	temp_list_retrained_models.append(retrained_model)
			# 	temp_list_lr.append(lr_retrain)
			# 	temp_list_lambda.append(lambda_retrain)
			# 	temp_list_CFAS.append(CFAS) 

			# if CFAS_flag == True:
			# 	for lr_retrain in [1e-5,1e-4,1e-3,1e-2]:
			# 		lambda_retrain = 0

			# 		retrained_model, CFAS = retrain(model, testloader, N_T_trainloader_c, N_T_testloader_c, device, fisher_dict, optpar_dict, num_retrain_epochs, lambda_retrain, lr_retrain, zeta)
					
			# 		# Append Data
			# 		temp_list_retrained_models.append(retrained_model)
			# 		temp_list_lr.append(lr_retrain)
			# 		temp_list_lambda.append(lambda_retrain)
			# 		temp_list_CFAS.append(CFAS) 

			if EWC_flag == True:
				for lr_retrain in [1e-5,1e-4,1e-3,1e-2]:
					for lambda_retrain in [0.25,0.5,0.75,1,2]:
						
						retrained_model, CFAS = retrain(model_multi, small_testloader_clas, N_T_trainloader_c_rot, N_T_testloader_c_rot, device, fisher_dict, optpar_dict, num_retrain_epochs, lambda_retrain, lr_retrain, zeta, layers_keywords=layers_keywords, fix_batch_noramlisation=fix_batch_noramlisation) 
						
						# Append Data
						temp_list_retrained_models.append(retrained_model)
						temp_list_lr.append(lr_retrain)
						temp_list_lambda.append(lambda_retrain)
						temp_list_CFAS.append(CFAS)     

			index_max = np.argmax(temp_list_CFAS)
			retrained_model = temp_list_retrained_models[index_max]
			CFAS = temp_list_CFAS[index_max]

			print("lr = ", temp_list_lr[index_max])
			print("lambda = ", temp_list_lambda[index_max])

			# Calculate accuracy of retrained model on target dataset X_tar 
			retrained_model.use_classification_head()
			_, A_T    = top1Accuracy(model=retrained_model, test_loader=testloader_c_clas, device=device, criterion=None)
			print("A_T = ",A_T)

			# best = fmin(fn=lambda x: retraining_objective(x),
			# 			space= {'x': [hp.uniform('ewc_lambda_hyper', 0, 100), hp.uniform('lr_retrain_hyper', 1e-5, 1e-2)]},
			# 			algo=tpe.suggest,
			# 			max_evals=100)

			# N_T_vs_A_T.append((N_T, A_T))
			results_log.append(noise_type, N_T, A_T,
								temp_list_lr[index_max], 
								temp_list_lambda[index_max], 
								zeta)
		
		# Log
		# logs_dic[noise_type] = N_T_vs_A_T

	results_log.write_file("exp"+str(experiment_number)+".txt")




if __name__ == "__main__":

	main()