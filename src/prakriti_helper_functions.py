import tensorflow as tf
from tensorflow.keras import backend as K
import joblib as jb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error

import os
import sys
sys.path.append("/content/mhcglobe/src")
import mhcglobe
import mhc_data
import inequality_loss
import train_functions as trainf
import binding_affinity as ba
import sequence_functions as seqf


def save_model(object_to_save, saved_object_filename):
  """
    Dumps object into a given directory.
    Note: This only works after the google drive is mounted in colab.

    Args:
      object_to_save: of any kind
      saved_object_filename: self-explanatory 

    Returns:
      None

    Use Case:
      from google.colab import drive
      drive.mount('/content/drive')
      
      saved_objects_path = "/content/drive/MyDrive/Colab Notebooks/MHCglobe/saved_objects/pMHC_data"
      jb.dump(pMHC_data, saved_objects_path)

    Note: You can then reload the object as follows:
      reloaded_pMHC_data = jb.load(saved_objects_path)

  """
  jb.dump(object_to_save, saved_object_filename)


def make_scatter_plot(df, x_col, y_col):
  plt.scatter(df[x_col], df[y_col])
  plt.xlabel(x_col)
  plt.ylabel(y_col)
  plt.title("Scatter Plot")
  plt.show()

def get_r_squared_mse(df, x_col, y_col):
  slope, intercept, r_value, p_value, std_err = linregress(df[x_col], df[y_col])
  mse = mean_squared_error(df[x_col], df[y_col])
  return(r_value**2, mse)

## Get the prediction of 1 model and compare it with true ##
def get_prediction_df(new_model, X, to_predict):
  """
    Gets the prediction of 1 model ("mhcglobe_affinities" "mhcglobe_scores") and concatenates with
      all information of the test_data (all other cols below)

    Args:
      new_model: loaded NN from Eric's ensemble (or any NN)
      X = matrix of features
      to_predict = test data set- dataframe with cols: "allele" "peptide" "measurement_value"

    Returns:
      prediction_df_all: 
        Dataframe with cols: "allele" "peptide" "measurement_inequality"  "measurement_value" "mhcglobe_affinities" "mhcglobe_scores"
    
    Use Case:
      new_model_path, verbose = "/content/mhcglobe/outputs", 0
      init_model = model_1

      new_model_1 = trainf.train_mhcglobe_model(init_model, X_tr, Y_tr, X_es, Y_es, new_model_path, verbose)
      X = seqf.get_XY(to_predict, encode_type="ONE_HOT", get_Y=False)
      to_predict = human_pMHC_data_test

      prediction_df_1 = get_prediction_df(new_model_1, X, to_predict)
  """
  mhcglobe_scores = new_model.predict(X)

  # Get them
  mhcglobe_scores = mhcglobe_scores.flatten()
  mhcglobe_affinities = list(map(ba.to_ic50, mhcglobe_scores))

  prediction_dict = {"mhcglobe_affinities": mhcglobe_affinities, "mhcglobe_scores": mhcglobe_scores}
  prediction_df = pd.DataFrame(prediction_dict)

  # Munge to_predict so I can correctly concatenate the columns
  munged_to_predict = to_predict
  munged_to_predict.index = prediction_df.index

  # Present the data nicely!
  prediction_df_all = pd.concat([munged_to_predict, prediction_df], axis=1)
  return(prediction_df_all)


def get_ensemble_predictions(new_mhcglobe_path, df_train, df_test, verbose):
  """
  Gets predictions for all models in new_mhcglobe_path.

  Args:
    df_train:
      Dataframe with cols: "allele" "peptide" "measurement_value"
      e.g. df_train = human_pMHC_data_test[["allele", "peptide", "measurement_value"]]

    new_mhcglobe_path: directory of where you want new trained models to go.
      Note: There will be an issue if you try to place trained models in a place
        where the same models already exist.

    df_test: test data with same structure as df_train
    verbose*: 1/0- with/without print statements

  Returns:
    ensemble_predictions:
      Dataframe with cols:
      (13, 14, ONE_HOT) (15, 37, ONE_HOT) (9, 79, ONE_HOT)  mhcglobe_score  mhcglobe_affinity
  
  Use Case:
    new_mhcglobe_path = "/content/mhcglobe/outputs/"
    df_train, df_test, verbose = pMHC_data_train, human_pMHC_data_test, 0
    ensemble_predictions = get_ensemble_predictions(new_mhcglobe_path, df_train, df_test, verbose)

  """
  # Make the initial ensemble
  init_ensemble = mhcglobe.ensemble(train_type='init', new_mhcglobe_path=new_mhcglobe_path)

  # Train all init models. They will appear in new_mhcglobe_path.
  new_model_list = list()
  new_model_paths = mhcglobe.LoadMHCGlobe().new_model_paths(new_mhcglobe_path)
  
  i = 0
  for init_model, new_model_path in zip(init_ensemble.ensemble_base_models, new_model_paths):
    print(f"Training model {i}")
    assert not os.path.exists(new_model_path), 'Already trained: {}'.format(new_model_path)
    # Data
    X_tr, Y_tr, X_es, Y_es = init_ensemble.setup_data_training(df_train)
    # Train
    new_model = trainf.train_mhcglobe_model(init_model, X_tr, Y_tr, X_es, Y_es, new_model_path, verbose)
    # Save them
    new_model_list.append(new_model)
    i += 1

  # Predict
  # Get X
  X = seqf.get_XY(df_test, encode_type=init_ensemble.protein_encoding, get_Y=False)

  ensemble_predictions = pd.DataFrame()
  base_model_predictions = []
  j = 0
  for model in new_model_list:
    print(f"Predicting for model {j}")
    predictions = pd.DataFrame(model.predict(X, verbose=0))
    base_model_predictions.append(predictions)
    j += 1

  ensemble_predictions = pd.concat(base_model_predictions, axis=1, ignore_index=True)
  ensemble_predictions.loc[:, 'mhcglobe_score'] = np.mean(ensemble_predictions, axis=1)
  ensemble_predictions.loc[:, 'mhcglobe_affinity'] = list(map(ba.to_ic50, ensemble_predictions['mhcglobe_score']))
  ensemble_predictions.columns = init_ensemble.hparam_ids + ['mhcglobe_score', 'mhcglobe_affinity']

  return ensemble_predictions

def make_comparison_df(ensemble_predictions, df_test):
  """
  Concatenates/compares "mhcglobe_affinity" calculated from ensemble_predictions and df_test.
  Note: mhcglobe_affinity is the average of mhcglobe_affinities calculated from
    the individual NNs in the ensemble.

  Args:
    ensemble_predictions (as above)
    df_test: test data- dataframe with cols: "allele" "peptide" "measurement_value"

  Returns:
    df_comparison:
      Dataframe with cols: "allele" "peptide" "measurement_value" "mhcglobe_affinity"
  
  Use Case:
    df_test = human_pMHC_data_test
    ensemble_predictions (from above)
    ensemble_comparison_df = make_comparison_df(ensemble_predictions, df_test)
  """
  ensemble_predictions.index = df_test.index
  df_comparison = pd.concat([df_test, ensemble_predictions], axis=1)
  df_comparison = df_comparison.loc[:, ['allele', 'peptide', 'measurement_value', 'mhcglobe_affinity']]
  return df_comparison

# useful for performance metrics during training
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

