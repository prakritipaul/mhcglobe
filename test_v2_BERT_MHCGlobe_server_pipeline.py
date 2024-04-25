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

## TO DO Make sure that imports are not circular! ##
import tensorflow as tf
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

## TO DO Download this into environment + export the neptune key ##
import neptune
from neptune.types import File

NEPTUNE_API_TOKEN = os.getenv("NEPTUNE_API_TOKEN")

# change
run = neptune.init_run(project="ppaul/tests",
    api_token=NEPTUNE_API_TOKEN,
    tags = ["server test"],
)

# # In case you have to access a previous run.

# run = neptune.init_run(project="ppaul/MHC-BERT",
#     api_token=NEPTUNE_API_TOKEN,
#     with_id="MHCBER-6",)

"""
Get data
Contains both BA+EL, human and non-human
"""
# 729,538/1,229,838
pMHC = mhc_data.pMHC_Data(only_EL=False, drop_duplicate_records=True)
pMHC_data = pMHC.data

"""
#### Randomly pick 100 HLA-peptide peptide + Remove them from training data
These will be the test set, and is the same number of examples as in Eric's example df.   
"""
# 678024
# Note: ba = 89529 (13%), sa/ma = 588495 (87%)
human_pMHC_data = pMHC_data[pMHC_data["allele"].str.contains("HLA")]
# 12% ba's
human_pMHC_data_test = human_pMHC_data.sample(n=100, random_state=428364)

# 729438
pMHC_data_train = pMHC_data[~pMHC_data.index.isin(human_pMHC_data_test.index)]

"""
#### Reset Indices to prevent any issues downstream
"""
pMHC_data_train = pMHC_data_train.reset_index(drop=True)
human_pMHC_data_test = human_pMHC_data_test.reset_index(drop=True)

"""
#### Make csv's that have all columns + save in Neptune

TO DO: Filepath/where to save
"""
new_directory = "../test"
os.makedirs(new_directory, exist_ok=True)

# Make a csv file that can be stored in neptune
pMHC_data_train.to_csv(new_directory + "testrun_train_428364.csv", index=False)
human_pMHC_data_test.to_csv(new_directory + "testrun_test_428364.csv", index=False)

# Neptune
run["data/all_columns_training"].upload(new_directory + "testrun_train_428364.csv")
run["data/all_columns_testing"].upload(new_directory + "testrun_test_428364.csv")

# """
# #### Get 4 relevant columns
# """
# pMHC_data_train = pMHC_data_train[["allele", "peptide", "measurement_inequality", "measurement_value"]]
# human_pMHC_data_test = human_pMHC_data_test[["allele", "peptide", "measurement_inequality", "measurement_value"]]

# """
# ### Balance the dataset to get X_tr, Y_tr, X_es, Y_es!
# """
# # Returns a data frame with cols
# # test (is it es or not), allele, peptide, measurement_inequality, measurement_value
# train, es = trainf.BalanceSplitData().get_train_val(pMHC_data_train)

# # reset indices to prevent downstream issues
# # 583650
# train = train.reset_index(drop=True)
# # 145788
# es = es.reset_index(drop=True)

# """
# #### Save these!
# TO DO: Filepath!
# """

# train_csv_name = "/content/mhcglobe/example/train_428364_24_04_11_BERT.csv"
# es_csv_name = "/content/mhcglobe/example/es_428364_24_04_11_BERT.csv"

# train.to_csv(train_csv_name, index=False)
# es.to_csv(es_csv_name, index=False)

# # Neptune
# run["data/train_428364_24_04_11_BERT"].upload(train_csv_name)
# run["data/es_428364_24_04_11_BERT"].upload(es_csv_name)
# # run["data/es_428364_24_04_05_es_1000"].upload(es_csv_name)

# """
# ### REMOVE THE INDICES LATER: just a test!
# """
# train, es = train.iloc[:1000], es.iloc[:500]


# """
# ### Get MHC pseudosxns (X1_train, X1_es)- should be a function
# Returns a list.
# """

# def get_allele_pseudoseqs(train_or_es, mhc_data):
#   """
#     Gets pseudosquences for MHC alleles present in a train or es dataframe.

#     Args:
#       train_or_es: Data frame with cols
#         test (is it es or not), allele, peptide, measurement_inequality, measurement_value

#       mhc data object: used to make allele2seq dict.
#         e.g. pMHC = mhc_data.pMHC_Data
#         {'HLA-A*02:560': 'YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYEWY'...}

#     Returns:
#       allele_pseudoseqs: list of shape (n_alleles, 34)
# "
#     Example:
#       'DLA-88*508:01', 'Mamu-B*08:01' -> 'YYATYGEKVETVYVDTLYITYRDYTWAVWNYTWY',
#                                          'YSSEYEERAGHTDADTLYLTYHYYTWAEVAYTWY'
#   """
#   mhc_alleles = train_or_es["allele"]
#   allele2seq_dict = mhc_data.allele2seq

#   allele_pseudoseqs = [str(allele2seq_dict[a]) for a in mhc_alleles]
#   return allele_pseudoseqs

# # X1_train, X1_es
# training_allele_pseudoseqs = get_allele_pseudoseqs(train, pMHC)
# es_allele_pseudoseqs = get_allele_pseudoseqs(es, pMHC)

# """
# ### Get MHCFlurry representations of peptides (X2_train, X2_es)
# Returns a list.

# Basically change "seq_to_15mer" function to just get the amino acids of the representation.
# """

# def get_mhcflurry_representation(peptide):
#   """
#     Given a peptide sequence, return the mhcflurry representation.

#     Examples:
#       Example 1: ARDV (4) -> ARDV-X7-ARDV
# 	    Example 2: ARDVA (5) -> ARDV-X7-RDVA
#       Example 3: ARDVAA (6) -> ARDV-X7-DVAA
#         (X7 padding is true until len(peptide) = 8)
#       Example 4: ARDVAAAAA (9) -> ARDV-XXX-A-XXX-AAAA
#       Example 5: ARDVAAAAAA (10) -> ARDV-XXX-AA-XX-AAAA
#   """
#   middle_AAs = peptide[4:-4]
#   num_X = 15-(len(middle_AAs)+8)

#   if num_X%2 == 0:
#     pad_len = num_X//2
#     middle_AAs_with_pad = "X"*pad_len + middle_AAs + "X"*pad_len

#   else:
#     pad_len_left = num_X//2 + 1
#     pad_len_right = pad_len_left - 1
#     middle_AAs_with_pad = "X"*pad_len_left + middle_AAs + "X"*pad_len_right

#   mhcflurry_representation = peptide[:4] + middle_AAs_with_pad + peptide[-4:]
#   return mhcflurry_representation

# def get_peptide_pseudoseqs(train_or_es):
#   """
#     Same function as "get_allele_pseudoseqs", but for peptides.

#     Returns:
#       peptide_pseudoseqs:
#         list of shape (n_peptides, 15)
#   """
#   peptides = train_or_es["peptide"]
#   peptide_pseudoseqs = [get_mhcflurry_representation(p) for p in peptides]
#   return peptide_pseudoseqs

# # X2_train, X2_es
# training_peptide_pseudoseqs = get_peptide_pseudoseqs(train)
# es_peptide_pseudoseqs = get_peptide_pseudoseqs(es)

# """
# ### Get BERT representation of both MHC and peptides (X1_train', X1_es', X2_train', X2_es')
# https://github.com/facebookresearch/esm/issues/348

# FUTURE IMPROVEMENT:
# Add a functionality where the attention mask is 0 wherever there's an X in the AA sxn (peptide).
# """

# def get_BERT_embeddings(aa_sxns, tokenizer, model, return_tensors):
#   """
#     Gets features of alleles or peptides from a BERT-like model, like ESM2.

#     Args:
#       aa_sxns: list of amino acid sequences.
#         e.g. training/es_allele_pseudoseqs, training_peptide/es_pseudoseqs

#       tokenizer: tokenizer for the model from huggingface.
#       model: model for the model from huggingface.
#       return_tensors: "tf" or "pt"

#     Returns:
#       PyTorch/Tensorflow tensors of shape n_alleles/peptides x 34/15 x n_embedding_dims
#         e.g. n_embedding_dims = 320 for ESM2_t6_8M_UR50D.
#         Note: embeddings are from the last hidden state of the model.

#     Example:
#       tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
#       model = TFEsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
#       get_BERT_embeddings(training_allele_pseudoseqs, tokenizer, model, "tf")
#   """
#   inputs = tokenizer(aa_sxns, return_tensors=return_tensors, padding=False, truncation=False)
#   input_ids = inputs["input_ids"]
#   attention_mask = inputs["attention_mask"]
#   # Update them to not have <cls> and <eos> tokens.
#   updated_input_ids = input_ids[:, 1:-1]
#   updated_attention_mask = attention_mask[:, 1:-1]
#   updated_inputs = {"input_ids": updated_input_ids, "attention_mask": updated_attention_mask}
#   # Get outputs
#   outputs = model(**updated_inputs)
#   # Get last_hidden_states
#   BERT_embeddings = outputs.last_hidden_state
#   return BERT_embeddings

# """
# #### TF version
# """
# from transformers import EsmTokenizer, TFEsmModel
# import torch

# ## TO DO: DOWNLOAD THE MODEL- REF DOCUMENTATION ## 
# tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
# model = TFEsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")

# # Neptune
# run["BERT/model"] = "facebook/esm2_t6_8M_UR50D"

# # Neptune
# run["BERT/num_training_samples"] = len(training_allele_pseudoseqs)
# run["BERT/num_es_samples"] = len(es_allele_pseudoseqs)


# # for test: 1000, 34, 320
# training_allele_BERT_embeddings = get_BERT_embeddings(training_allele_pseudoseqs, tokenizer, model, "tf")
# training_peptide_BERT_embeddings = get_BERT_embeddings(training_peptide_pseudoseqs, tokenizer, model, "tf")

# es_allele_BERT_embeddings = get_BERT_embeddings(es_allele_pseudoseqs, tokenizer, model, "tf")
# es_peptide_BERT_embeddings = get_BERT_embeddings(es_peptide_pseudoseqs, tokenizer, model, "tf")


# """
# ### Get Y from get_XY function
# """
# Y_tr = seqf.get_XY(train, encode_type='ONE_HOT', get_Y=True)[1]
# Y_es = seqf.get_XY(es, encode_type='ONE_HOT', get_Y=True)[1]


# """
# ### Make new model with dense layer
# """

# # TO DO: CHANGE this path (doesn't matter because it never gets used)
# new_mhcglobe_path = "/content/mhcglobe/example/"
# init_ensemble = mhcglobe.ensemble(train_type='init', new_mhcglobe_path=new_mhcglobe_path)

# model_3 = init_ensemble.ensemble_base_models[2]

# """
# #### Let's look at it!
# """

# # CIRCULAR IMPORT?
# from tensorflow.keras.utils import plot_model
# plot_model(model_3, to_file='model.png', show_shapes=True, show_layer_names=True)

# # Neptune
# run["model/model_tf_graph"].upload("/content/model.png")

# # Neptune
# run["model/eric_model_dir"] = model_dir

# # Define input tensors
# BERT_allele = Input(shape=(34, 320))
# peptide_allele = Input(shape=(15, 320))

# # Dimensionality reduction layers
# reduced_allele = Dense(20, kernel_regularizer = tf.keras.regularizers.l1(l=0.01), activation='relu')(BERT_allele)
# reduced_peptide = Dense(20, kernel_regularizer = tf.keras.regularizers.l1(l=0.01), activation='relu')(peptide_allele)

# # Assume model takes two inputs: [reduced_input_1, reduced_input_2]
# # Adjusted to reshape if necessary and connect to model
# output_1 = model([reduced_allele, reduced_peptide])

# # Define the new comprehensive model
# new_model = Model(inputs=[BERT_allele, peptide_allele], outputs=output_1)

# # Neptune
# BERT_architecture = {"embedding_layer": "last",
#                      "layers": "2 Dense- separate dim reduction",
#                      "kernel_regularizer": "l1(l=0.01)",
#                      "activation": "relu",
#                      "inputs": "BERT_allele (30, 320); peptide_allele (15, 320)",
#                      "output": "model on dim reduction of BERT features = 20 dims"}

# run["BERT/BERT_architecture"] = BERT_architecture

# """
# ### Compilation related
# """

# # Neptune
# # model 1
# # model_optimizer_params = {"learning_rate": 0.0011339304,
# #                             "momentum": 0.5,
# #                             "epsilon": 6.848580326162904e-07,
# #                             "centered": True,
# #                             "optimizer_type": "RMSprop"}

# model_3_optimizer_params = {"optimizer_type": "RMSprop",
#                             "learning_rate": 0.0019147476,
#                             "momentum": 0.5,
#                             "epsilon": 3.17051703095139e-07,
#                             "centered": True}

# run["model/optimizer_params"] = model_3_optimizer_params

# model_optimizer_params = model_3_optimizer_params

# optimizer = optimizers.RMSprop(
#     learning_rate=model_optimizer_params["learning_rate"],
#     momentum=model_optimizer_params["momentum"],
#     epsilon=model_optimizer_params["epsilon"],
#     centered=model_optimizer_params["centered"])

# # Neptune
# model_compiler_params = {"optimizer": "RMSprop",
#                            "loss": "inequality_loss.MSEWithInequalities().loss",
#                            "metrics": ["mean_absolute_error", "mean_squared_error", "root_mean_squared_error"]}

# # Converts list into correct format.
# model_compiler_params["metrics"] = json.dumps(model_compiler_params["metrics"])
# run["model/compiler_params"] = model_compiler_params

# """
# ### Compile the model
# Code is not ideal
# """

# new_model.compile(optimizer=optimizer, loss=inequality_loss.MSEWithInequalities().loss, metrics=["mean_absolute_error", "mean_squared_error", phf.root_mean_squared_error])

# """
# ### Check what this `new_model` is like!
# """
# new_model.summary()

# from tensorflow.keras.utils import plot_model
# plot_model(new_model, to_file='model.png', show_shapes=True, show_layer_names=True)

# # Neptune
# from neptune.types import File
# run["model/tf_graph"].upload("/content/model.png")

# """
# ### Train the model
# """

# # Neptune
# training_specs = {"batch_size": 100,
#                   "epochs": 300,
#                   "shuffle": True,
#                   "verbose": 1,
#                   "mhc_callbacks": ["monitor=val_loss", "patience=20", "mode=min", "baseline=1", "min_delta=0.0001"],
#                   "early_stopping": True,
#                   "num_training_samples": 1000,
#                   "num_es_samples": 500}

# # Converts list into correct format.
# training_specs["mhc_callbacks"] = json.dumps(training_specs["mhc_callbacks"])

# # Neptune
# run["training/training_specs"] = training_specs

# mhcglobe_callbacks = [Callbacks.EarlyStopping(
#                 monitor='val_loss',
#                 patience=20,
#                 mode='min',
#                 baseline=1,
#                 min_delta=0.0001)]

# """
# ### Training!
# """
# verbose = 1
# history = new_model.fit([training_allele_BERT_embeddings, training_peptide_BERT_embeddings], Y_tr,
#                         batch_size= training_specs["batch_size"], #hparams['batch_size'], 10000
#                         epochs=training_specs["epochs"],
#                         validation_data=([es_allele_BERT_embeddings, es_peptide_BERT_embeddings], Y_es),
#                         shuffle=True,
#                         verbose=verbose,
#                         callbacks=mhcglobe_callbacks)

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
#     run["training/loss"].append(loss[i])
#     run["training/val_loss"].append(val_loss[i])
#     run["training/mae"].append(mae[i])
#     run["training/val_mae"].append(val_mae[i])
#     run["training/mse"].append(mse[i])
#     run["training/val_mse"].append(val_mse[i])
#     run["training/rmse"].append(rmse[i])
#     run["training/val_rmse"].append(val_rmse[i])

# """
# ### Predict using this model!!!
# """

# test_allele_pseudoseqs = get_allele_pseudoseqs(human_pMHC_data_test, pMHC)
# test_peptide_pseudoseqs = get_peptide_pseudoseqs(human_pMHC_data_test)

# # 100, 34, 320
# test_allele_BERT_embeddings = get_BERT_embeddings(test_allele_pseudoseqs, tokenizer, model, "tf")
# test_peptide_BERT_embeddings = get_BERT_embeddings(test_peptide_pseudoseqs, tokenizer, model, "tf")

# # Only has mhcglobe scores.
# test_predictions = new_model.predict([test_allele_BERT_embeddings, test_peptide_BERT_embeddings])

# mhcglobe_scores = new_model.predict([test_allele_BERT_embeddings, test_peptide_BERT_embeddings])

# # Get them
# mhcglobe_scores = mhcglobe_scores.flatten()
# mhcglobe_affinities = list(map(ba.to_ic50, mhcglobe_scores))

# prediction_dict = {"mhcglobe_affinities": mhcglobe_affinities, "mhcglobe_scores": mhcglobe_scores}
# prediction_df = pd.DataFrame(prediction_dict)

# # Munge to_predict so I can correctly concatenate the columns
# munged_to_predict = human_pMHC_data_test
# munged_to_predict.index = prediction_df.index

# # Present the data nicely!
# prediction_df_all = pd.concat([munged_to_predict, prediction_df], axis=1)

# """
# ### Save the predictions
# """
# run["testing/test_predictions_428364_24_04_11_BERT"].upload(File.as_pickle(test_predictions))
# run["testing/test_predictions_df_all_24_04_11_BERT"].upload(File.as_pickle(prediction_df_all))
# run["testing/test_predictions_df_all_viz_428364_24_04_11_BERT"].upload(File.as_html(prediction_df_all))
# """
# ### Get Scatter Plot + r^2/MSE
# """
# import matplotlib.pyplot as plt

# savefig_name = "scatter_plot_BERT_model3_train1000_es500_24_04_11.png"

# plt.scatter(prediction_df_all["measurement_value"], prediction_df["mhcglobe_affinities"])
# plt.xlabel("measurement_value")
# plt.ylabel("mhcglobe_affinities")
# plt.title("Scatter Plot")
# plt.savefig(savefig_name)

# predictions_r, predictions_mse = phf.get_r_squared_mse(prediction_df_all, "measurement_value", "mhcglobe_affinities")
# print("R-squared:", predictions_r)
# print("MSE:", predictions_mse)

# # Neptune
# run["testing/r_squared"] = predictions_r
# run["testing/mse"] = predictions_mse
# run["testing/scatter_plot"].upload((savefig_name))

# # end neptune
# run.stop()
