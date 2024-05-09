"""
		- Tensorflow version of running OneHot pipeline (adapted from Eric's code).
		- Goal is to compare OneHot results using Eric's base model 3 to
			my PyTorch code (long-term implementation).
		- Will use random seed + Eric's example test.
		- BERT + base model modifications will be done in PyTorch.

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
import joblib as jb
import pickle
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
sys.path.append("../src")
import mhcglobe
import mhc_data
import inequality_loss
import parameters as params
import train_functions as trainf
import binding_affinity as ba
import sequence_functions as seqf
import prakriti_helper_functions as phf

def get_train_test_data(pMHC, sample_num=None, random_state=None, train_data_csv=None, test_data_csv=None, makedir=False, out_dir=None, out_train_data_csv=None, out_test_data_csv=None):
		"""
		- Given pMHC_data (all data in MHCGlobe = human + human', BA + EL)
		- Get subsets = train + test samples
		
		Args:

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
			* All columns are present
		"""
		pMHC_data = pMHC.data
		human_pMHC_data = pMHC_data[pMHC_data["allele"].str.contains("HLA")]
		
		# both training and testing data are provided.
		if train_data_csv != None and test_data_csv != None:
			train_data = pd.read_csv(train_data_csv)
			test_data = pd.read_csv(test_data_csv)
			
		else:
			# in the case of uploading Eric's example.
			if test_data_csv != None:
				human_pMHC_data_test = pd.read_csv(test_data_csv)
			else:
				human_pMHC_data_test = human_pMHC_data.sample(n=sample_num, random_state=random_state)

			pMHC_data_train = pMHC_data[~pMHC_data.index.isin(human_pMHC_data_test.index)]
			train_data = pMHC_data_train.reset_index(drop=True)
			test_data = human_pMHC_data_test.reset_index(drop=True)
		
		# makes out_dir.
		if makedir:
			if not os.path.exists(out_dir):
				os.makedirs(out_dir, exist_ok=True)
				train_data.to_csv(out_dir + out_train_data_csv, index=False)
				test_data.to_csv(out_dir + out_test_data_csv, index=False)
			else:
				train_data.to_csv(out_dir + out_train_data_csv, index=False)
				test_data.to_csv(out_dir + out_test_data_csv, index=False)


		return train_data, test_data


def get_train_es_data(train_data_df=None, train_data_csv=None, es_data_csv=None, makedir=False, out_dir=None, out_train_data2_csv=None, out_es_data_csv=None):
	"""
	- Takes training data from above and splits it into train + es data.
	- Features only 4 relevant cols: allele, peptide, measurement_inequality, measurement_value
	- Splits training data into train and es. 

	Args:
		- train_data_df: training data from above function
	
	Returns:
		- train + es
		- Also writes them out to outdir.
		* Cols:  test (True/False), allele, peptide measurement_inequality, measurement_value
	"""
	# dataframe is provided; e.g. output of above function.
	if train_data_df is not None:
		train_data_df = train_data_df[["allele", "peptide", "measurement_inequality", "measurement_value"]] 
		train, es = trainf.BalanceSplitData().get_train_val(train_data_df)
		train = train.reset_index(drop=True)
		es = es.reset_index(drop=True)
	
	# training and es data are provided.
	elif train_data_csv != None and es_data_csv != None:
		train = pd.read_csv(train_data_csv)
		es = pd.read_csv(es_data_csv)
	
	if makedir:
		# make out_dir if it doesn't exist
		if not os.path.exists(out_dir):
			os.makedirs(out_dir, exist_ok=True)
			train.to_csv(out_dir + out_train_data2_csv, index=False)
			es.to_csv(out_dir + out_es_data_csv, index=False)

		else:
			train.to_csv(out_dir + out_train_data2_csv, index=False)
			es.to_csv(out_dir + out_es_data_csv, index=False)
	
	return train, es

def get_OneHot_features_Y(train=None, es=None, test=None, train_csv=None, es_csv=None, test_csv=None):
	"""
	- gets OneHot encodings + Y values
	- returns lists
	"""
	
	if train_csv != None:
		train = pd.read_csv(train_csv)
	if es_csv != None:
		es = pd.read_csv(es_csv)
	if test_csv != None:
		test = pd.read_csv(test_csv)
	
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
		encode_type="ONE_HOT",
		get_Y=True)
	
	return X_train, X_es, X_test, Y_train, Y_es, Y_test

def get_model(model_num):
	# model 3 -> index 2
	new_mhcglobe_path = "/content/mhcglobe/example/"
	init_ensemble = mhcglobe.ensemble(train_type='init', new_mhcglobe_path=new_mhcglobe_path)
	model = init_ensemble.ensemble_base_models[model_num-1]
	return model

def get_model_optimizer_params(model_num):
	# get the optimizer params and return a dict
	# this will be useful for future neptune calls
	return params.optimizer_params[model_num]

def recompile_model(model):
	model.compile(optimizer=model.optimizer,
			   loss=model.loss,
			   metrics=["mean_absolute_error", "mean_squared_error", phf.root_mean_squared_error]
			   )

def make_training_specs(batch_size, epochs, shuffle, verbose, num_training_samples, num_es_samples):
	# makes a dict using user args.
 	# will be useful for future neptune calls
	training_specs = {"batch_size": batch_size,
					"epochs": epochs,
					"shuffle": epochs,
					"verbose": verbose,
					"mhc_callbacks": ["monitor=val_loss", "patience=20", "mode=min", "baseline=1", "min_delta=0.0001"],
					"early_stopping": True,
					"num_training_samples": num_training_samples,
					"num_es_samples": num_es_samples}

	# Converts list into correct format.
	training_specs["mhc_callbacks"] = json.dumps(training_specs["mhc_callbacks"])
	return training_specs
	
def train_model(model, X_train, Y_train, X_es, Y_es, training_specs):
	# X_train and X_es are OneHot-encoded features of train and es respectively.
	# Y_train and Y_es are the binding affinities.
	history = model.fit(X_train, Y_train,
                        batch_size= training_specs["batch_size"], #hparams['batch_size'], 10000
                        epochs=training_specs["epochs"],
                        validation_data=(X_es, Y_es),
                        shuffle=training_specs["shuffle"],
                        verbose=training_specs["verbose"],
                        callbacks=params.mhcglobe_callbacks)
	return history

def get_training_performance_metrics(history):
	training_performance_metrics = {"loss": history.history['loss'],
									"val_loss": history.history['val_loss'],
									"mae": history.history['mean_absolute_error'],
									"val_mae": history.history['val_mean_absolute_error'],
									"mse": history.history['mean_squared_error'],
									"val_mse": history.history['val_mean_squared_error'],
									"rmse": history.history['root_mean_squared_error'],
									"val_rmse": history.history['val_root_mean_squared_error']}
	return training_performance_metrics

if __name__ == "__main__":
	# args
	model_num = 3
	batch_size, epochs, shuffle, verbose, num_training_samples, num_es_samples = 10000, 300, True, 1, "all", "all"
	# change
	fig_name = "Model 3 Scatter Plot"
	# change
	pickle_name = "server_OneHot_run_24_05_09.pkl"
	
	# global var
	pMHC = mhc_data.pMHC_Data(only_EL=False, drop_duplicate_records=True)

	# input data files : local
	# data_dir = "/Users/prakritipaul/Library/CloudStorage/GoogleDrive-ppaul@alumni.princeton.edu/My Drive/Pre-industry Professional Prep/AImmunology/server_related/output_files/BERT_colab_run_outputs_24_05_01/"
	# train_data_csv = data_dir + "all_columns_pMHC_data_train_428364.csv"
	# test_data_csv = data_dir + "all_columns_human_pMHC_data_test_428364.csv"

	# della
	DATA_DIR = "/home/ppaul/mhcglobe_test_files/OneHot_colab_run_outputs_24_05_08/"
	all_train_data_csv = DATA_DIR + "all_columns_train_seed-428364_24_05_08.csv"
	test_data_csv = DATA_DIR + "all_columns_test_seed-428364_24_05_08.csv"
	train_data_csv = DATA_DIR + "train_seed-428364_24_05_08.csv"
	es_data_csv = DATA_DIR + "es_seed-428364_24_05_08.csv"

	# local: If you wanted to write out these files
	# out_dir = "/Users/prakritipaul/Library/CloudStorage/GoogleDrive-ppaul@alumni.princeton.edu/My Drive/Pre-industry Professional Prep/AImmunology/server_related/output_files/25_05_07-VS-BERT_colab_run_outputs_24_05_08/"
	out_dir = "/home/ppaul/mhcglobe_test_files/"
	# out_train_data_csv = "test_train_25_05_07b.csv"
	# out_test_data_csv = "test_test_25_05_07b.csv"

	#############

	# Pipeline
	print("Getting training and test data")
	train_data, test_data = get_train_test_data(pMHC, train_data_csv=all_train_data_csv, test_data_csv=test_data_csv)
	print("Getting training and validation data")
	train, es = get_train_es_data(train_data_csv=train_data_csv, es_data_csv=es_data_csv)
	print("Getting OneHot features")
	X_train, X_es, X_test, Y_train, Y_es, Y_test = get_OneHot_features_Y(train, es, test_data)

	print("Getting model")
	model = get_model(model_num)
	optimizer_params = get_model_optimizer_params(model_num)
	print("Recompiling it")
	recompile_model(model)
	
	print("Training Model")
	training_specs = make_training_specs(batch_size, epochs, shuffle, verbose, num_training_samples, num_es_samples)
	history = train_model(model, X_train, Y_train, X_es, Y_es, training_specs)
	training_performance_metrics = get_training_performance_metrics(history)
	print("Testing Model")
	prediction_df = phf.get_prediction_df(model, X_test, test_data)
	print("Getting Test Performance Metrics")
	scatter_plot = phf.make_scatter_plot(prediction_df, "measurement_value", "mhcglobe_affinities", fig_name, out_dir)
	predictions_r, predictions_mse = phf.get_r_squared_mse(prediction_df, "measurement_value", "mhcglobe_affinities")

	print("Pickle everything")
	pickle_dict = {"optimizer_params": optimizer_params,
				"training_specs": training_specs,
				"history": history,
				"training_performance_metrics": training_performance_metrics,
				"prediction_df": prediction_df,
				"predictions_r": predictions_r,
				"predictions_mse": predictions_mse}

	with open(out_dir+pickle_name, 'wb') as f:
		pickle.dump(pickle_dict, f)