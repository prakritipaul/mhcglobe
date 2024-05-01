"""
		Tensorflow version of running the BERT pipeline

		TO DO:
				- DONE: Move functions to phf!
				
				to generalize:
				- Have a way of picking the model! (model_1)
				- facebook model
				- scatter plot name
				
				- DONE: Maybe change variable name model -> BERT_model
				- DONE: Fix circular imports
						- from tensorflow.keras.utils import plot_model

				- DONE: Make a dictionary with the params of both model_1 and model_3- then you pick.
				- DONE: In fact, I can make a separate script that has those.

				Basically, generalize this code!
"""
import numpy as np
import pandas as pd
import joblib as jb ###
import json
import matplotlib.pyplot as plt

## TO DO Make sure that imports are not circular! ##
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.callbacks as Callbacks
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers, losses

## TO DO Need to figure out correct filepath for this ##
import os
import sys
sys.path.append("./src")
import mhcglobe
import mhc_data
import inequality_loss
import train_functions as trainf
import binding_affinity as ba
import sequence_functions as seqf
import prakriti_helper_functions as phf


def get_train_test_data(pMHC, train_test_type, sample_num=None, random_state=None, train_data_csv=None, test_data_csv=None, out_dir, out_train_data_csv, out_test_data_csv):
		"""
		- Given pMHC_data (all data in MHCGlobe = human + human', BA + EL)
		- Get subsets = train + test samples
		
		Args:
			1) train_test_type:
				- human:
					- 80/20 train/test Human HLA split
						Note: This is the specific split because of
							the trainf.BalanceSplitData().get_train_val function.
				- human_non_human:
					- 100 randomly sampled human samples (test), rest (train)

			2) out_dir
				- New directory where you want following csv's to be saved.
				-  This directory is created.
				e.g. out_dir = "../test"

			3) out_train_csv_name
				- Name of csv that will contain training data

			4) out_test_csv_name
			- Name of csv that will contain testing data
		
		Returns:
			1) train_data
				Note: train_data will include both training + validation sets.
			2) test_data
			3) 2 csvs will be written to out_dir.
			
		"""
		pMHC_data = pMHC.data
		human_pMHC_data = pMHC_data[pMHC_data["allele"].str.contains("HLA")]
		
		if train_test_type == "human":
				human_train_data, human_test_data = trainf.BalanceSplitData().get_train_val(human_pMHC_data)
				
				human_pMHC_data_train = human_train_data[[col for col in human_train_data.columns if col != 'test']]
				human_pMHC_data_test = human_test_data[[col for col in human_test_data.columns if col != 'test']]

				train_data = human_pMHC_data_train.reset_index(drop=True)
				test_data = human_pMHC_data_test.reset_index(drop=True)

		elif train_test_type == "human_non_human":
			if train_data_csv != None and test_data_csv != None:
				train_data = pd.read_csv(train_data_csv)
				test_data = pd.read_csv(test_data_csv)
				
			else:
				if test_data_csv != None:
					human_pMHC_data_test = pd.read_csv(test_data_csv)

				elif sample_num != None and random_state != None:
					human_pMHC_data_test = human_pMHC_data.sample(n=sample_num, random_state=random_state)

				pMHC_data_train = pMHC_data[~pMHC_data.index.isin(human_pMHC_data_test.index)]
				train_data = pMHC_data_train.reset_index(drop=True)
				test_data = human_pMHC_data_test.reset_index(drop=True)

		# Make out_dir
		os.makedirs(out_dir, exist_ok=True)
		# Make csv's that have all columns
		train_data.to_csv(out_dir + out_train_data_csv, index=False)
		test_data.to_csv(out_dir + out_test_data_csv, index=False)

		return train_data, test_data


def get_train_es_data(train_data_df=None, train_data_csv=None, es_data_csv=None, out_dir, out_train_data2_csv, out_es_data_csv):
	"""
	- Takes training data from above and splits it into train + es data.
	- Features only 4 relevant cols: allele, peptide, measurement_inequality, measurement_value
	- Splits training data into train and es. 

	Args:
		- train_data_df: training data from above function
	
	Returns:
		- train + es
		- Also writes them out to outdir.
	"""
	train_data_df = train_data_df[["allele", "peptide", "measurement_inequality", "measurement_value"]] 
	if train_data_df != None:
		train, es = trainf.BalanceSplitData().get_train_val(train_data_df)
		train = train.reset_index(drop=True)
		es = es.reset_index(drop=True)
	
	elif train_data_csv != None and es_data_csv != None:
		train = pd.read_csv(train_data_csv)
		es = pd.read_csv(es_data_csv)
	
	if not os.path.exists(out_dir):
		os.makedirs(out_dir, exist_ok=True)
		train.to_csv(out_dir + out_train_data2_csv, index=False)
		es.to_csv(out_dir + out_es_data_csv, index=False)

	else:
		train.to_csv(out_dir + out_train_data2_csv, index=False)
		es.to_csv(out_dir + out_es_data_csv, index=False)
	
	return train, es


def get_BERT_features_Y(train, es, test, pMHC, tokenizer, model, model_type="tf"):
	"""
		- First gets their allele pseudoseqs.
		- Gets peptide pseudoseqs of train, es, and test
		- Then gets BERT embeddings
		- Also returns Y
		- We get embeddings for both alleles and peptides because
			we will concatenate them down the line.
	"""
	train_allele_pseudoseqs = phf.get_allele_pseudoseqs(train, pMHC)
	train_peptide_pseudoseqs = phf.get_peptide_pseudoseqs(train)
	train_allele_BERT_embeddings = phf.get_BERT_embeddings(train_allele_pseudoseqs, tokenizer, model, "tf")
	train_peptide_BERT_embeddings = phf.get_BERT_embeddings(train_peptide_pseudoseqs, tokenizer, model, "tf")
	Y_train = seqf.get_XY(train, encode_type='ONE_HOT', get_Y=True)[1]
	
	es_allele_pseudoseqs = phf.get_allele_pseudoseqs(es, pMHC)
	es_peptide_pseudoseqs = phf.get_peptide_pseudoseqs(es)
	es_allele_BERT_embeddings = phf.get_BERT_embeddings(es_allele_pseudoseqs, tokenizer, model, "tf")
	es_peptide_BERT_embeddings = phf.get_BERT_embeddings(es_peptide_pseudoseqs, tokenizer, model, "tf")
	Y_es = seqf.get_XY(es, encode_type='ONE_HOT', get_Y=True)[1]

	test_allele_pseudoseqs = phf.get_allele_pseudoseqs(test, pMHC)
	test_peptide_pseudoseqs = phf.get_peptide_pseudoseqs(test)
	test_allele_BERT_embeddings = phf.get_BERT_embeddings(test_allele_pseudoseqs, tokenizer, model, "tf")
	test_peptide_BERT_embeddings = phf.get_BERT_embeddings(test_peptide_pseudoseqs, tokenizer, model, "tf")
	Y_test = seqf.get_XY(es, encode_type='ONE_HOT', get_Y=True)[1]

	return train_peptide_BERT_embeddings, es_peptide_BERT_embeddings, test_peptide_BERT_embeddings, Y_train, Y_es, Y_test

def get_OneHot_features_Y(train, es, test):
	"""
		- Same as above, just for OneHot
	"""
	X_train, Y_train = seqf.get_XY(
		train,
		encode_type="ONE_HOT",
		get_Y=True)

	X_es, Y_es = seqf.get_XY(
		es,
		encode_type="ONE_HOT",
		get_Y=True)
	
	X_test, Y_test = seqf.get_XY(
		test,
		encode_type="ONE_HOT"
		get_Y=True)
	
	return X_train, X_es, X_test, Y_train, Y_es, Y_test
	

"""
### Make new model with dense layer
"""

# TO DO: CHANGE this path (doesn't matter because it never gets used)
new_mhcglobe_path = "/content/mhcglobe/example/"
init_ensemble = mhcglobe.ensemble(train_type='init', new_mhcglobe_path=new_mhcglobe_path)

model_3 = init_ensemble.ensemble_base_models[2]

"""
#### Let's look at it!
"""

# CIRCULAR IMPORT?
plot_model(model_3, to_file='model.png', show_shapes=True, show_layer_names=True)

# Define input tensors
BERT_allele = Input(shape=(34, 320))
peptide_allele = Input(shape=(15, 320))

# Dimensionality reduction layers
reduced_allele = Dense(20, kernel_regularizer = tf.keras.regularizers.l1(l=0.01), activation='relu')(BERT_allele)
reduced_peptide = Dense(20, kernel_regularizer = tf.keras.regularizers.l1(l=0.01), activation='relu')(peptide_allele)

# Assume model takes two inputs: [reduced_input_1, reduced_input_2]
# Adjusted to reshape if necessary and connect to model
output_1 = model([reduced_allele, reduced_peptide])

# Define the new comprehensive model
new_model = Model(inputs=[BERT_allele, peptide_allele], outputs=output_1)

# # Neptune
# BERT_architecture = {"embedding_layer": "last",
# 										 "layers": "2 Dense- separate dim reduction",
# 										 "kernel_regularizer": "l1(l=0.01)",
# 										 "activation": "relu",
# 										 "inputs": "BERT_allele (30, 320); peptide_allele (15, 320)",
# 										 "output": "model on dim reduction of BERT features = 20 dims"}

# run["BERT/BERT_architecture"] = BERT_architecture

"""
### Compilation related
"""

# Neptune
# model 1
# model_optimizer_params = {"learning_rate": 0.0011339304,
#                             "momentum": 0.5,
#                             "epsilon": 6.848580326162904e-07,
#                             "centered": True,
#                             "optimizer_type": "RMSprop"}

# model_3_optimizer_params = {"optimizer_type": "RMSprop",
# 														"learning_rate": 0.0019147476,
# 														"momentum": 0.5,
# 														"epsilon": 3.17051703095139e-07,
# 														"centered": True}

# run["model/optimizer_params"] = model_3_optimizer_params

model_optimizer_params = model_3_optimizer_params

optimizer = optimizers.RMSprop(
		learning_rate=model_optimizer_params["learning_rate"],
		momentum=model_optimizer_params["momentum"],
		epsilon=model_optimizer_params["epsilon"],
		centered=model_optimizer_params["centered"])

# # Neptune
# model_compiler_params = {"optimizer": "RMSprop",
# 													 "loss": "inequality_loss.MSEWithInequalities().loss",
# 													 "metrics": ["mean_absolute_error", "mean_squared_error", "root_mean_squared_error"]}

# # Converts list into correct format.
# model_compiler_params["metrics"] = json.dumps(model_compiler_params["metrics"])
# run["model/compiler_params"] = model_compiler_params

"""
### Compile the model
Code is not ideal
"""

new_model.compile(optimizer=optimizer, loss=inequality_loss.MSEWithInequalities().loss, metrics=["mean_absolute_error", "mean_squared_error", phf.root_mean_squared_error])

"""
### Check what this `new_model` is like!
"""
new_model.summary()

plot_model(new_model, to_file='model.png', show_shapes=True, show_layer_names=True)

# # Neptune
# from neptune.types import File
# run["model/tf_graph"].upload("/content/model.png")

"""
### Train the model
"""

# # Neptune
# training_specs = {"batch_size": 100,
# 									"epochs": 300,
# 									"shuffle": True,
# 									"verbose": 1,
# 									"mhc_callbacks": ["monitor=val_loss", "patience=20", "mode=min", "baseline=1", "min_delta=0.0001"],
# 									"early_stopping": True,
# 									"num_training_samples": 1000,
# 									"num_es_samples": 500}

# # Converts list into correct format.
# training_specs["mhc_callbacks"] = json.dumps(training_specs["mhc_callbacks"])

# # Neptune
# run["training/training_specs"] = training_specs

mhcglobe_callbacks = [Callbacks.EarlyStopping(
								monitor='val_loss',
								patience=20,
								mode='min',
								baseline=1,
								min_delta=0.0001)]

"""
### Training!
"""
verbose = 1
history = new_model.fit([train_allele_BERT_embeddings, train_peptide_BERT_embeddings], Y_tr,
												batch_size= training_specs["batch_size"], #hparams['batch_size'], 10000
												epochs=training_specs["epochs"],
												validation_data=([es_allele_BERT_embeddings, es_peptide_BERT_embeddings], Y_es),
												shuffle=True,
												verbose=verbose,
												callbacks=mhcglobe_callbacks)

# """
# ### Save with Neptune!
# """
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# mae = history.history['mean_absolute_error']
# val_mae = history.history['val_mean_absolute_error']
# mse = history.history['mean_squared_error']
# val_mse = history.history['val_mean_squared_error']
# rmse = history.history['root_mean_squared_error']
# val_rmse = history.history['val_root_mean_squared_error']

# for i in range(len(loss)):
# 		run["training/loss"].append(loss[i])
# 		run["training/val_loss"].append(val_loss[i])
# 		run["training/mae"].append(mae[i])
# 		run["training/val_mae"].append(val_mae[i])
# 		run["training/mse"].append(mse[i])
# 		run["training/val_mse"].append(val_mse[i])
# 		run["training/rmse"].append(rmse[i])
# 		run["training/val_rmse"].append(val_rmse[i])

"""
### Predict using this model!!!
"""

test_allele_pseudoseqs = phf.get_allele_pseudoseqs(human_pMHC_data_test, pMHC)
test_peptide_pseudoseqs = phf.get_peptide_pseudoseqs(human_pMHC_data_test)

# 100, 34, 320
test_allele_BERT_embeddings = phf.get_BERT_embeddings(test_allele_pseudoseqs, tokenizer, model, "tf")
test_peptide_BERT_embeddings = phf.get_BERT_embeddings(test_peptide_pseudoseqs, tokenizer, model, "tf")

# Only has mhcglobe scores.
test_predictions = new_model.predict([test_allele_BERT_embeddings, test_peptide_BERT_embeddings])

mhcglobe_scores = new_model.predict([test_allele_BERT_embeddings, test_peptide_BERT_embeddings])

# Get them
mhcglobe_scores = mhcglobe_scores.flatten()
mhcglobe_affinities = list(map(ba.to_ic50, mhcglobe_scores))

prediction_dict = {"mhcglobe_affinities": mhcglobe_affinities, "mhcglobe_scores": mhcglobe_scores}
prediction_df = pd.DataFrame(prediction_dict)

# Munge to_predict so I can correctly concatenate the columns
munged_to_predict = human_pMHC_data_test
munged_to_predict.index = prediction_df.index

# Present the data nicely!
prediction_df_all = pd.concat([munged_to_predict, prediction_df], axis=1)

"""
### Save the predictions
"""
run["testing/test_predictions_428364_24_04_11_BERT"].upload(File.as_pickle(test_predictions))
run["testing/test_predictions_df_all_24_04_11_BERT"].upload(File.as_pickle(prediction_df_all))
run["testing/test_predictions_df_all_viz_428364_24_04_11_BERT"].upload(File.as_html(prediction_df_all))
"""
### Get Scatter Plot + r^2/MSE
"""
savefig_name = "scatter_plot_BERT_model3_train1000_es500_24_04_11.png"

plt.scatter(prediction_df_all["measurement_value"], prediction_df["mhcglobe_affinities"])
plt.xlabel("measurement_value")
plt.ylabel("mhcglobe_affinities")
plt.title("Scatter Plot")
plt.savefig(savefig_name)

predictions_r, predictions_mse = phf.get_r_squared_mse(prediction_df_all, "measurement_value", "mhcglobe_affinities")
print("R-squared:", predictions_r)
print("MSE:", predictions_mse)

# # Neptune
# run["testing/r_squared"] = predictions_r
# run["testing/mse"] = predictions_mse
# run["testing/scatter_plot"].upload((savefig_name))

# # end neptune
# run.stop()

if __name__ == "__main__":
	# arg: human/human_nonhuman
	train_test_type = None
	# other args:
 	sample_num, random_state, train_data_csv, test_data_csv, out_dir, out_train_data_csv, out_train_data2_csv, out_test_data_csv, out_es_data_csv = None, None, None, None, None, None, None, None, None

	# arg: OneHot/BERT
	feature_type = None
	# args if BERT:
	tokenizer, model, model_type = None, None, "tf"

	from transformers import EsmTokenizer, TFEsmModel
	import torch

	## TO DO: DOWNLOAD THE MODEL- REF DOCUMENTATION ## 
	tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
	model = TFEsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
	
	#####
	
	# global var
	pMHC = mhc_data.pMHC_Data(only_EL=False, drop_duplicate_records=True)

	# pipeline
	train_data, test = get_train_test_data(pMHC, train_test_type, sample_num=None, random_state=None, train_data_csv=None, test_data_csv=None, out_dir, out_train_data_csv, out_test_data_csv):
	train, es = get_train_es_data(train_data_df=None, train_data_csv=None, es_data_csv=None, out_dir, out_train_data2_csv, out_es_data_csv)

	if feature_type == "BERT":
		train_peptide_BERT_embeddings, es_peptide_BERT_embeddings, test_peptide_BERT_embeddings, Y_train, Y_es, Y_test = get_BERT_features_Y(train, es, test, pMHC, tokenizer, model, model_type="tf")
		
		# Training
 
		# Testing	
	
	elif feature_type == "OneHot":
		X_train, X_es, X_test, Y_train, Y_es, Y_test = get_OneHot_features_Y(train, es, test)

		# Training
 
		# Testing
 
	# Metrics