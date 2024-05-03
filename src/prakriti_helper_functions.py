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


def make_scatter_plot(df, x_col, y_col, fig_name, save_dir):
  plt.scatter(df[x_col], df[y_col])
  plt.xlabel(x_col)
  plt.ylabel(y_col)
  plt.title("Scatter Plot")
  plt.savefig(save_dir+fig_name)
  return plt

def get_r_squared_mse(df, x_col, y_col):
  slope, intercept, r_value, p_value, std_err = linregress(df[x_col], df[y_col])
  mse = mean_squared_error(df[x_col], df[y_col])
  return(r_value**2, mse)

## Get the prediction of 1 model and compare it with true ##
def get_prediction_df(new_model, X, test_data):
  """
    Gets the prediction of 1 model ("mhcglobe_affinities" "mhcglobe_scores") and concatenates with
      all information of the test_data (all other cols below)

    Args:
      new_model: loaded NN from Eric's ensemble (or any NN)
      X = matrix of features of test data set.
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
  munged_to_predict = test_data
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

########### useful for performance metrics during training ###########
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

########### For BERT ###########
def get_allele_pseudoseqs(train_or_es, mhc_data):
  """
    Gets pseudosquences for MHC alleles present in a train or es dataframe.

    Args:
      train_or_es: Data frame with cols
        test (is it es or not), allele, peptide, measurement_inequality, measurement_value

      mhc data object: used to make allele2seq dict.
        e.g. pMHC = mhc_data.pMHC_Data
        {'HLA-A*02:560': 'YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYEWY'...}

    Returns:
      allele_pseudoseqs: list of shape (n_alleles, 34)
"
    Example:
      'DLA-88*508:01', 'Mamu-B*08:01' -> 'YYATYGEKVETVYVDTLYITYRDYTWAVWNYTWY',
                                         'YSSEYEERAGHTDADTLYLTYHYYTWAEVAYTWY'
  """
  mhc_alleles = train_or_es["allele"]
  allele2seq_dict = mhc_data.allele2seq

  allele_pseudoseqs = [str(allele2seq_dict[a]) for a in mhc_alleles]
  return allele_pseudoseqs

def get_mhcflurry_representation(peptide):
  """
    Given a peptide sequence, return the mhcflurry representation.

    Examples:
      Example 1: ARDV (4) -> ARDV-X7-ARDV
      Example 2: ARDVA (5) -> ARDV-X7-RDVA
      Example 3: ARDVAA (6) -> ARDV-X7-DVAA
        (X7 padding is true until len(peptide) = 8)
      Example 4: ARDVAAAAA (9) -> ARDV-XXX-A-XXX-AAAA
      Example 5: ARDVAAAAAA (10) -> ARDV-XXX-AA-XX-AAAA
  """
  middle_AAs = peptide[4:-4]
  num_X = 15-(len(middle_AAs)+8)

  if num_X%2 == 0:
    pad_len = num_X//2
    middle_AAs_with_pad = "X"*pad_len + middle_AAs + "X"*pad_len

  else:
    pad_len_left = num_X//2 + 1
    pad_len_right = pad_len_left - 1
    middle_AAs_with_pad = "X"*pad_len_left + middle_AAs + "X"*pad_len_right

  mhcflurry_representation = peptide[:4] + middle_AAs_with_pad + peptide[-4:]
  return mhcflurry_representation

def get_peptide_pseudoseqs(train_or_es):
  """
    Same function as "get_allele_pseudoseqs", but for peptides.
    
    Returns:
      peptide_pseudoseqs:
        list of shape (n_peptides, 15)
  """
  peptides = train_or_es["peptide"]
  peptide_pseudoseqs = [get_mhcflurry_representation(p) for p in peptides]
  return peptide_pseudoseqs

def get_BERT_embeddings(aa_sxns, tokenizer, model, return_tensors):
  """
    Gets features of alleles or peptides from a BERT-like model, like ESM2.

    Args:
      aa_sxns: list of amino acid sequences.
        e.g. training/es_allele_pseudoseqs, training_peptide/es_pseudoseqs

      tokenizer: tokenizer for the model from huggingface.
      model: model for the model from huggingface.
      return_tensors: "tf" or "pt"

    Returns:
      PyTorch/Tensorflow tensors of shape n_alleles/peptides x 34/15 x n_embedding_dims
        e.g. n_embedding_dims = 320 for ESM2_t6_8M_UR50D.
        Note: embeddings are from the last hidden state of the model.

    Example:
      tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
      model = TFEsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
      get_BERT_embeddings(training_allele_pseudoseqs, tokenizer, model, "tf")
  """
  inputs = tokenizer(aa_sxns, return_tensors=return_tensors, padding=False, truncation=False)
  input_ids = inputs["input_ids"]
  attention_mask = inputs["attention_mask"]
  # Update them to not have <cls> and <eos> tokens.
  updated_input_ids = input_ids[:, 1:-1]
  updated_attention_mask = attention_mask[:, 1:-1]
  updated_inputs = {"input_ids": updated_input_ids, "attention_mask": updated_attention_mask}
  # Get outputs
  outputs = model(**updated_inputs)
  # Get last_hidden_states
  BERT_embeddings = outputs.last_hidden_state
  return BERT_embeddings


