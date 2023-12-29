import os
import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd


# =======================================
# Samples VS Accuracy (CAS vs no CAS)
# =======================================

# experiment_id_array = [106, 104 ,105,109]#108] # resnet-26 (CIFAR-10C) DIRA with CAS
experiment_id_array = [30, 38]#108] # resnet-26 (CIFAR-10C) DIRA without CAS
# experiment_id_array = [115,116, 117, 118] # resnet-26 (CIFAR-100C) DIRA with CAS

# Source_acuracy = 0#0.637#0.504

for experiment_id in experiment_id_array:

	file_name = "Results_logs/exp"+str(experiment_id)+".txt"
	results_folder = "Results_plots"

	df = pd.read_csv(file_name)
	NOISE_TYPES = np.unique(df["noise_type"])
	# NOISE_TYPES = ["contrast", "zoom_blur", "defocus_blur", "fog"]
	first_run_flag = True
	for noise_type in NOISE_TYPES:
		df_extracted  = df[df.loc[:,"noise_type"]==noise_type]
		N_T_array = df_extracted["N_T"].values

		if first_run_flag:
			A_T_array = df_extracted["A_T"].values
			first_run_flag = False
		else:
			A_T_array += df_extracted["A_T"].values
			
	A_T_array = A_T_array/len(NOISE_TYPES)
	Source_acuracy = A_T_array[0]

	# print(Source_acuracy)
	# input("press enter to proceed")

	# A_T_array = np.insert(A_T_array,0, Source_acuracy)
	# N_T_array = np.insert(N_T_array,0, 0)

	if experiment_id == 38:
		label_name = "DIRA-SS (with CAS)"
		color_name = "red"
		linestyle  = "-"
	
	if experiment_id == 30:
		label_name = "DIRA-SS ($\eta$ = 1e-5, $\lambda$ = 1)"
		color_name = "blue"
		linestyle  = "-"
	
	elif experiment_id == 34:
		label_name = "SGD ($\eta$ = 1e-5)"
		color_name = "orange"
		linestyle  = "-"

	plt.plot(N_T_array, A_T_array*100, label=label_name, color = color_name, linestyle=linestyle)

plt.plot([0,202], [Source_acuracy*100, Source_acuracy*100], label="Source", linestyle="--", color="grey")


plt.legend()
plt.ylim(0,100)
# plt.xlim(0,200)
plt.xlim(0,100)
# plt.xlim(0,50)
plt.xlabel("Number of Samples for Adaptation")
plt.ylabel("Top-1 Classification Accuracy (%)")
plt.savefig(results_folder+"/CAS_VS_NO_CAS_CIFAR10C_Resnet26_N_TvsA_T.pdf")
# input("wait")

# =======================================
# Layers
# =======================================
plt.clf()
# experiment_id_array = [106, 104 ,105,109]#108] # resnet-26 (CIFAR-10C) DIRA with CAS
experiment_id_array = [30, 31, 32, 33]#, 34, 35, 36, 37]#108] # resnet-26 (CIFAR-10C) DIRA without CAS
# experiment_id_array = [115,116, 117, 118] # resnet-26 (CIFAR-100C) DIRA with CAS

# Source_acuracy = 0#0.637#0.504

for experiment_id in experiment_id_array:

	file_name = "Results_logs/exp"+str(experiment_id)+".txt"
	results_folder = "Results_plots"

	df = pd.read_csv(file_name)
	NOISE_TYPES = np.unique(df["noise_type"])
	# NOISE_TYPES = ["contrast", "zoom_blur", "defocus_blur", "fog"]
	first_run_flag = True
	for noise_type in NOISE_TYPES:
		df_extracted  = df[df.loc[:,"noise_type"]==noise_type]
		N_T_array = df_extracted["N_T"].values

		if first_run_flag:
			A_T_array = df_extracted["A_T"].values
			first_run_flag = False
		else:
			A_T_array += df_extracted["A_T"].values
			
	A_T_array = A_T_array/len(NOISE_TYPES)
	Source_acuracy = A_T_array[0]

	# print(Source_acuracy)
	# input("press enter to proceed")

	# A_T_array = np.insert(A_T_array,0, Source_acuracy)
	# N_T_array = np.insert(N_T_array,0, 0)

	if experiment_id == 30:
		label_name = "Block 1,2,3,4" #"DIRA-SS (output)"
		color_name = "blue"
		linestyle  = "-"
	
	elif experiment_id == 31:
		label_name = "Block 1,2,3"
		color_name = "blue"
		linestyle  = "--"

	elif experiment_id == 32:
		label_name = "Block 1,2"
		color_name = "blue"
		linestyle  = "-."

	elif experiment_id == 33:
		label_name = "Block 1"
		color_name = "blue"
		linestyle  = "dotted"
	
	elif experiment_id == 34:
		label_name = "SGD (output)"
		color_name = "orange"
		linestyle  = "-"

	elif experiment_id == 35:
		label_name = "SGD (from block 4)"
		color_name = "orange"
		linestyle  = "--"

	elif experiment_id == 36:
		label_name = "SGD (from block 3)"
		color_name = "orange"
		linestyle  = "-."

	elif experiment_id == 37:
		label_name = "SGD (from block 2)"
		color_name = "orange"
		linestyle  = "dotted"

	plt.plot(N_T_array, A_T_array*100, label=label_name, color = color_name, linestyle=linestyle)

plt.plot([0,202], [Source_acuracy*100, Source_acuracy*100], label="Source", linestyle="--", color="grey")


plt.legend()
plt.ylim(0,100)
plt.ylim(40,80)
# plt.xlim(0,200)
plt.xlim(0,50)
# plt.xlim(0,50)
plt.xlabel("Number of Samples for Adaptation")
plt.ylabel("Top-1 Classification Accuracy (%)")
plt.savefig(results_folder+"/Layers_CIFAR10C_Resnet26_N_TvsA_T.pdf")
# input("wait")


# =======================================
# Dyanmic Adaptation Scenario
# =======================================
# experiment_id_array = [66, 64]


# Source_acuracy = [0.270, 0.593, 0.510, 0.625] # strictly in the order of "contrast","defocus_blur", "fog", "zoom_blur"
# Baseline_accuracy = 0.855

experiment_id = 32#121#109


file_name = "Results_logs/exp"+str(experiment_id)+".txt"
results_folder = "Results_plots"

df = pd.read_csv(file_name)
NOISE_TYPES = np.unique(df["noise_type"])

first_run_flag = True
last_N_T = 0

plt. clf()
# plt.plot([0, 600], [Baseline_accuracy*100, Baseline_accuracy*100], label="Baseline CIFAR-10", linestyle="--", color="grey")

counter = -1
for noise_type in ["contrast","defocus_blur", "fog", "zoom_blur"]:
	counter += 1
	if noise_type == "original":
		A_T_array    = np.array([Baseline_accuracy, Baseline_accuracy])
		N_T_array    = np.array([0,100])
		

	else:

		df_extracted  = df[df.loc[:,"noise_type"]==noise_type]
		A_T_array = df_extracted["A_T"].values
		N_T_array = df_extracted["N_T"].values

		# A_T_array = A_T_array[:5] # np.insert(A_T_array[:5],0, Source_acuracy)
		# N_T_array = N_T_array[:5] - 2# np.insert(N_T_array[:5],0, 0) -2

		# A_T_array = np.insert(A_T_array[:5], 0, Source_acuracy[counter])
		# N_T_array = np.insert(N_T_array[:5],0, 0) 
		# N_T_array[-1] = 100
	
	A_T_array = A_T_array *100

	if first_run_flag:
		pass
	else:
		# A_T_array = np.insert(A_T_array,0, A_T_array_old[-1]) 
		# N_T_array = np.insert(N_T_array,0, N_T_array_old[0])
		pass
	
	A_T_array_old    = A_T_array
	N_T_array_old    = N_T_array 

	N_T_array = N_T_array + last_N_T
	last_N_T  = N_T_array[-1]

	if first_run_flag:
		first_run_flag = False
		plt.plot(N_T_array, A_T_array, color = "blue", label="DIRA ")
		plt.plot([N_T_array[-1], N_T_array[-1]], [0,100], linestyle="--", color = "green", linewidth = '0.5', label="Change of Domain")
	else:
		plt.plot(N_T_array, A_T_array, color = "blue")
		plt.plot([N_T_array[-1], N_T_array[-1]], [0,100], linestyle="--", color = "green", linewidth = '0.5')

	if noise_type == "original":
		text = "Baseline"
		x = N_T_array[0]+15

	elif noise_type == "contrast":
		text = "Contrast"
		x = N_T_array[0]+20

	elif noise_type == "defocus_blur":
		text = "Defocus Blur"
		x = N_T_array[0]+15
	
	elif noise_type == "fog":
		text = "Fog"
		x = N_T_array[0]+35

	elif noise_type == "zoom_blur":
		text = "Zoom Blur"
		x = N_T_array[0]+20

	elif noise_type == "original":
		text = "Baseline"
		x = N_T_array[0]+10

	plt.text(x,30, text)
	plt.text(N_T_array[0]+5,25, str(round(A_T_array[0], 1))+"% -> "+str(round(A_T_array[-1], 1))+"%")#, size=7)



plt.legend()
plt.ylim(0,100)
plt.xlim(0,400)
plt.xlabel("Number of Samples")
plt.ylabel("Top-1 Classification Accuracy (%)")
plt.savefig(results_folder+"/DynamicScenario.pdf")


first_run_flag = True
last_N_T = 0

plt. clf()
# plt.plot([0, 600], [Baseline_accuracy*100, Baseline_accuracy*100], label="Baseline CIFAR-10", linestyle="--", color="grey")

counter = -1
for noise_type in NOISE_TYPES:#["contrast","defocus_blur", "fog", "zoom_blur"]:
	counter += 1
	if noise_type == "original":
		A_T_array    = np.array([Baseline_accuracy, Baseline_accuracy])
		N_T_array    = np.array([0,100])
		

	else:

		df_extracted  = df[df.loc[:,"noise_type"]==noise_type]
		A_T_array = df_extracted["A_T"].values
		N_T_array = df_extracted["N_T"].values

		# A_T_array = A_T_array[:5] # np.insert(A_T_array[:5],0, Source_acuracy)
		# N_T_array = N_T_array[:5] - 2# np.insert(N_T_array[:5],0, 0) -2

		# A_T_array = np.insert(A_T_array[:5], 0, Source_acuracy[counter])
		# N_T_array = np.insert(N_T_array[:5],0, 0) 
		# N_T_array[-1] = 100
	
	A_T_array = A_T_array *100

	if first_run_flag:
		pass
	else:
		# A_T_array = np.insert(A_T_array,0, A_T_array_old[-1]) 
		# N_T_array = np.insert(N_T_array,0, N_T_array_old[0])
		pass
	
	A_T_array_old    = A_T_array
	N_T_array_old    = N_T_array 

	N_T_array = N_T_array + last_N_T
	last_N_T  = N_T_array[-1]

	if first_run_flag:
		first_run_flag = False
		plt.plot(N_T_array, A_T_array, color = "blue", label="DIRA ")
		plt.plot([N_T_array[-1], N_T_array[-1]], [0,100], linestyle="--", color = "green", linewidth = '0.5', label="Change of Domain")
	else:
		plt.plot(N_T_array, A_T_array, color = "blue")
		plt.plot([N_T_array[-1], N_T_array[-1]], [0,100], linestyle="--", color = "green", linewidth = '0.5')

	x = N_T_array[0]+15
	noise_type_modified = noise_type.replace('_', '\n')

	plt.text(x,30, noise_type_modified, fontsize=3)
	# if len(noise_type) > 10:
	# 	# Split the text into two lines
	# 	lines = noise_type.split('\n')
		
	# 	# Add text with reduced font size for each line
	# 	plt.text(x, 30, lines[0], fontsize=3, ha='left', va='bottom')
		
	# 	if len(lines) > 1:
	# 		plt.text(x, 25, lines[1], fontsize=3, ha='left', va='top')
	# else:
	# 	# Add text with reduced font size for short text
	# 	plt.text(x, 30, noise_type, fontsize=3, ha='left', va='bottom')


	# plt.text(N_T_array[0]+5,25, str(round(A_T_array[0], 1))+"% -> "+str(round(A_T_array[-1], 1))+"%")#, size=7)

		

plt.legend()
plt.ylim(0,100)
plt.xlim(0,1500)
plt.xlabel("Number of Samples")
plt.ylabel("Top-1 Classification Accuracy (%)")
plt.savefig(results_folder+"/DynamicScenario_all.pdf")




# =======================================
# Samples order do not matter
# =======================================

experiment_id = 39#121#109


file_name = "Results_logs/exp"+str(experiment_id)+".txt"
results_folder = "Results_plots"

directory = "Results_logs"
file_prefix = "exp"+str(experiment_id)
file_list = [file for file in os.listdir(directory) if file.startswith(file_prefix)]

data_frames = [pd.read_csv(os.path.join(directory, file), index_col=1) for file in file_list]

# Ensure all dataframes have the same columns and index
for i in range(1, len(data_frames)):
    if not data_frames[i - 1].columns.equals(data_frames[i].columns) or \
       not data_frames[i - 1].index.equals(data_frames[i].index):
        print(f"Warning: Dataframes in files {file_list[i - 1]} and {file_list[i]} do not have the same structure.")

# concatenated_df = pd.concat(data_frames, axis=1)

# Extract the column "A_T" from each dataframe
column_data = [df["A_T"] for df in data_frames]
concatenated_column_df = pd.concat(column_data, axis=1)*100


print(concatenated_column_df)
print(concatenated_column_df[concatenated_column_df.index==5])

averaged_concatenated_column_df = concatenated_column_df[concatenated_column_df.index==0].mean(axis=0).to_frame().T
averaged_concatenated_column_df.index = [0]
temp_df                         = concatenated_column_df[concatenated_column_df.index==5].mean(axis=0).to_frame().T
temp_df.index = [5]
averaged_concatenated_column_df = averaged_concatenated_column_df.append(temp_df)


# averaged_concatenated_column_df = concatenated_column_df[concatenated_column_df.index==5].mean(axis=0).to_frame().T
# averaged_concatenated_column_df.index = [5]

temp_df                         = concatenated_column_df[concatenated_column_df.index==100].mean(axis=0).to_frame().T
temp_df.index = [100]

averaged_concatenated_column_df = averaged_concatenated_column_df.append(temp_df)


print(averaged_concatenated_column_df)
# input("wait")
 # Calculate mean and std for each row in the "A_T" column
row_mean = averaged_concatenated_column_df.mean(axis=1)
row_std = averaged_concatenated_column_df.std(axis=1)

print(row_mean)
print(row_std)

# Display the results
result_df = pd.DataFrame({'Mean_A_T': row_mean, 'Std_A_T': row_std})



# Plotting error bar diagram
plt.clf()
plt.figure(figsize=(10, 6))
plt.errorbar(result_df.index, result_df['Mean_A_T'], yerr=result_df['Std_A_T'], fmt='o', label='Mean with Std')

selected_ticks = [5, 100]
plt.xticks(selected_ticks)

plt.xlabel("Number of Samples")
plt.ylabel("Top-1 Classification Accuracy (%)")
# plt.ylim(0,100)
plt.xlim(-50,150)
plt.grid(True)
plt.savefig(results_folder+"/samples_type_effect_error.pdf")


# For Box and whisker diagram
plt.clf()
plt.figure(figsize=(8, 6))
averaged_concatenated_column_df.T.boxplot()#column=['5 ', '100'])
plt.xlabel("Number of Samples")
plt.ylabel("Top-1 Classification Accuracy (%)")
# plt.ylim(40,80)
plt.grid(True)
plt.savefig(results_folder+"/samples_type_effect_box.pdf")

