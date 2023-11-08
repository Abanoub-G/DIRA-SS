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
from utils.dataloaders import augmented_samples_dataloader_iterative, augmented_samples_dataloader_iterative_imagenet
from utils.dataloaders import auxilary_samples_dataloader_iterative

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


DATASET_DIR              = '../datasets/CIFAR10/'#'../../NetZIP/datasets/TinyImageNet/' #'../datasets/CIFAR100/' # '../../NetZIP/datasets/ImageNet/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC' 
DATASET_NAME             = "CIFAR10" # Options: "CIFAR10" "CIFAR100" "TinyImageNet"  "ImageNet"

NUM_CLASSES              = 1000 # Number of classes in dataset

MODEL_CHOICE             = "resnet" # Option:"resnet" "vgg"
MODEL_VARIANT            = "resnet18" # Common Options: "resnet18" "vgg11" For more options explore files in models to find the different options.

MODEL_DIR                = "../models/" + MODEL_CHOICE
MODEL_SELECTION_FLAG     = 2 # create an untrained model = 0, start from a pytorch trained model = 1, start from a previously saved local model = 2

MODEL_FILENAME     = MODEL_VARIANT +"_"+DATASET_NAME+".pt"
MODEL_FILEPATH     = os.path.join(MODEL_DIR, MODEL_FILENAME)


NOISE_TYPES_ARRAY = ["brightness","contrast","defocus_blur",
					"elastic_transform","fog","frost","gaussian_blur",
					"gaussian_noise", "glass_blur", "impulse_noise",
					"jpeg_compression", "motion_blur", "pixelate", 
					"saturate", "shot_noise", "snow", "spatter", 
					"speckle_noise", "zoom_blur"]

NOISE_TYPES_ARRAY = ["gaussian_noise"]#,"shot_noise"]

# NOISE_TYPES_ARRAY = ["jpeg_compression", "motion_blur", "pixelate", "shot_noise", "snow", "zoom_blur"]
# NOISE_TYPES_ARRAY = ["contrast","motion_blur","fog"]

# NOISE_TYPES_ARRAY = ["impulse_noise"]

NOISE_SEVERITY 	  = 5 # Options from 1 to 5

MAX_SAMPLES_NUMBER = 120 #220
N_T_Initial        = 1
N_T_STEP = 25 #16

def retrain(model, testloader, N_T_trainloader_c, N_T_testloader_c, device, fisher_dict, optpar_dict, num_retrain_epochs, lambda_retrain, lr_retrain, zeta):
	
	model.use_rotation_head()
	# model.use_classification_head()

	# Copy model for retraining
	retrained_model = copy.deepcopy(model)

	# retrained_model.use_classification_head()
	retrained_model.use_rotation_head()
	
	# Retrain
	retrained_model = train_model_ewc(model = retrained_model, 
									train_loader = N_T_trainloader_c, 
									test_loader = N_T_testloader_c, 
									device = device, 
									fisher_dict = fisher_dict,
									optpar_dict = optpar_dict,
									num_epochs=num_retrain_epochs, 
									ewc_lambda = lambda_retrain,
									learning_rate=lr_retrain, 
									momentum=0.9, 
									weight_decay=1e-5)


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


	# Calculate CFAS
	# CFAS = A_k.cpu().numpy() * (zeta*A_0.cpu().numpy() +1)
	CFAS = A_k.cpu().numpy() + 10*A_0.cpu().numpy()
	# CFAS = A_k.cpu().numpy() + 2*A_0.cpu().numpy()
	# CFAS = A_k.cpu().numpy() + 1.5*A_0.cpu().numpy()
	# CFAS = A_k.cpu().numpy() * (5*A_0.cpu().numpy() +1)
	# CFAS = A_k.cpu().numpy() * (10*A_0.cpu().numpy() +1)

	return retrained_model, CFAS

def main():

	# ========================================
	# == Preliminaries
	# ========================================
	# Fix seeds to allow for repeatable results 
	set_random_seeds(SEED_NUMBER)

	# Setup device used for training either gpu or cpu
	device = set_cuda(USE_CUDA)

	# Load model
	model = model_selection(model_selection_flag=MODEL_SELECTION_FLAG, model_dir=MODEL_DIR, model_choice=MODEL_CHOICE, model_variant=MODEL_VARIANT, saved_model_filepath=MODEL_FILEPATH, num_classes=NUM_CLASSES, device=device, mutli_selection_flag = False)
	model_multi = model_selection(model_selection_flag=MODEL_SELECTION_FLAG, model_dir=MODEL_DIR, model_choice=MODEL_CHOICE, model_variant=MODEL_VARIANT, saved_model_filepath=MODEL_FILEPATH, num_classes=NUM_CLASSES, device=device, mutli_selection_flag = True)
	# model_multi_copy = copy.deepcopy(model_multi)
	print("Progress: Model has been setup.")

	# Setup original dataset
	trainloader_clas, testloader_clas,_ = pytorch_dataloader(dataset_name=DATASET_NAME, dataset_dir=DATASET_DIR, images_size=32, batch_size=64)
	trainloader_rot, testloader_rot = pytorch_rotation_dataloader(dataset_name=DATASET_NAME, dataset_dir=DATASET_DIR, images_size=32, batch_size=64)
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

	# for param in model_multi.resnet.layer4.parameters():
	# 	param.requires_grad = True

	# for param in model_multi.resnet.avgpool.parameters():
	# 	param.requires_grad = True

	# print("Classificaiton")
	# for param in model_multi.classification_head.parameters():#resnet.fc.parameters():
	# 	print(param)
	# 	input("press enter to continue")
	# 	param.requires_grad = True

	# print("Rotation")
	# flag_temp = True
	# for param in model_multi.rotation_head.parameters():#resnet.fc.parameters():
		# print(param)
		# if flag_temp:
		# 	flag_temp= False
		# 	continue
		# input("press enter to continue")
		# param.requires_grad = True

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

	# print("Classificaiton")
	# for param in model_multi.classification_head.parameters():#resnet.fc.parameters():
	# 	print(param)
	# 	input("press enter to continue")
	# 	param.requires_grad = True

	# print("Rotation")
	# flag_temp = True
	# for param in model_multi.rotation_head.parameters():#resnet.fc.parameters():
	# 	print(param)
		# input("press enter to continue")
		# param.requires_grad = True

	# model_multi.use_rotation_head()
	# model_multi_copy.use_rotation_head()
	# print("================ Change in Rotation head")
	# for (name1, param1), (name2, param2) in zip(model_multi_copy.named_parameters(), model_multi.named_parameters()):
	# 	if param1.size() == param2.size():
	# 		diff = torch.abs(param1 - param2)
			# print(f"Parameter Name: {name1}")
			# print(f"Max Absolute Difference: {diff.max()}")
			# print(f"Min Absolute Difference: {diff.min()}")
			# print(f"Mean Absolute Difference: {diff.mean()}")
			# print("-----")
	
	# print("================ Change in Classificaiton head")
	# model_multi.use_classification_head()
	# model_multi_copy.use_classification_head()
	# for (name1, param1), (name2, param2) in zip(model_multi_copy.named_parameters(), model_multi.named_parameters()):
	# 	if param1.size() == param2.size():
	# 		diff = torch.abs(param1 - param2)
			# print(f"Parameter Name: {name1}")
			# print(f"Max Absolute Difference: {diff.max()}")
			# print(f"Min Absolute Difference: {diff.min()}")
			# print(f"Mean Absolute Difference: {diff.mean()}")
			# print("-----")

	
	# Stopped at running this script

	# STOPEED AT LOOKING INTO:
	# 	- Fine Tuning the new head parameters based on the rotation dataset, without changing the main model parameters
	# 	- Reassess the model after this fintuning. The classificaiton head should be the same but the rotation one might improve.
	# 	- You can try to fine tune the main parameters as well using a low learning rate to improve accuracy on rotation detection.
	# 	- Retrain on new domain based on rotation task only. Measure accuracy. 

	# # Set model to appropriate head
	# model_multi.use_rotation_head()

	# Set model to classificaton head
	model_multi.use_classification_head()

	# Initiate dictionaries for regularisation using EWC	
	fisher_dict = {}
	optpar_dict = {}

	optimizer = optim.SGD(model_multi.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
	
	save_dict = False
	load_dict = True

	if load_dict == True and (DATASET_NAME == "ImageNet" or DATASET_NAME == "TinyImageNet"):
		with open("resnet18_"+DATASET_NAME+"_fisher.pkl", 'rb') as file:
			fisher_dict = pickle.load(file)

		with open("resnet18_"+DATASET_NAME+"_optpar.pkl", 'rb') as file:
			optpar_dict = pickle.load(file)

		print("PROGRESS: Calculated Fisher loaded")

	else:
		# fisher_dict, optpar_dict = on_task_update(0, trainloader_clas, model, optimizer, fisher_dict, optpar_dict, device)
		fisher_dict, optpar_dict = on_task_update(0, trainloader_clas, model_multi, optimizer, fisher_dict, optpar_dict, device)
		print("PROGRESS: Calculated Fisher matrix")

		if save_dict == True and (DATASET_NAME == "ImageNet" or DATASET_NAME == "TinyImageNet"):

			with open("resnet18_"+DATASET_NAME+"_fisher.pkl", 'wb') as file:
				pickle.dump(fisher_dict, file)

			with open("resnet18_"+DATASET_NAME+"_optpar.pkl", 'wb') as file:
				pickle.dump(optpar_dict, file)

			print("PROGRESS: Saved Fisher matrix")



	# ========================================
	# == Load Noisy Data
	# ========================================
	logs_dic = {}
	results_log = logs() 

	for noise_type in NOISE_TYPES_ARRAY:
		if noise_type == "original":
			testloader_c = testloader
		else:
			# load noisy dataset
			if DATASET_NAME == "CIFAR10" or DATASET_NAME == "CIFAR100":
				trainloader_c_clas, testloader_c_clas, noisy_images, noisy_labels = cifar_c_dataloader(NOISE_SEVERITY, noise_type, DATASET_NAME)

			elif DATASET_NAME == "ImageNet":
				trainloader_c, testloader_c, train_set, test_set     = imagenet_c_dataloader(NOISE_SEVERITY, noise_type)

			elif DATASET_NAME == "TinyImageNet":
				trainloader_c, testloader_c,  train_set, test_set    = imagenet_c_dataloader(NOISE_SEVERITY, noise_type, tiny_imagenet=True)
			# print("shape of noisy images = ", np.shape(noisy_images))
			# print("shape of noisy labels = ", np.shape(noisy_labels))
		
		# Evaluate testloader_c on original model
		model_multi.use_classification_head()
		_, eval_accuracy_clas_multi   = top1Accuracy(model=model_multi, test_loader=testloader_c_clas, device=device, criterion=None)
		# model_multi.use_rotation_head()
		# _, eval_accuracy_rot_multi    = top1Accuracy(model=model_multi, test_loader=testloader_c_rot, device=device, criterion=None)
		
		print("Cls Accuray on " + noise_type +" dataset = ",eval_accuracy_clas_multi)
		# print("Rot Accuray on " + noise_type +" dataset = ",eval_accuracy_rot_multi)

		# model_multi.use_classification_head()
		# _,eval_accuracy     = top1Accuracy(model=model, test_loader=testloader_c_clas, device=device, criterion=None)
		# original_model_eval_accuracy       = eval_accuracy.cpu().numpy()
		# print(noise_type +" dataset = ",original_model_eval_accuracy)
		

		# ========================================	
		# == Select random N_T number of datapoints from noisy data for retraining
		# ========================================
		N_T_vs_A_T = []
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
			# elif DATASET_NAME == "ImageNet" or DATASET_NAME == "TinyImageNet":
			# 	N_T_trainloader_c, N_T_testloader_c, samples_indices_array = augmented_samples_dataloader_iterative_imagenet(N_T, train_set, test_set, samples_indices_array, N_T_STEP)



			# ========================================	
			# == Retrain model
			# ========================================
			num_retrain_epochs = 10

			zeta = 1

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
						# model_multi.use_rotation_head()
						# model_multi.use_classification_head()
						retrained_model, CFAS = retrain(model_multi, testloader_clas, N_T_trainloader_c_rot, N_T_testloader_c_rot, device, fisher_dict, optpar_dict, num_retrain_epochs, lambda_retrain, lr_retrain, zeta) 
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

			N_T_vs_A_T.append((N_T, A_T))
			results_log.append(noise_type, N_T, A_T,
								temp_list_lr[index_max], 
								temp_list_lambda[index_max], 
								zeta)
		
		# Log
		logs_dic[noise_type] = N_T_vs_A_T

	results_log.write_file("exp1.txt")




if __name__ == "__main__":

	main()