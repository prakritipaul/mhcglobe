import numpy as np
import pandas as pd
import joblib as jb
import json

print("Importing tf")
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.callbacks as Callbacks
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers, losses

print("Importing mhcglobe files")
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

# Get data
print("Getting data")
pMHC = mhc_data.pMHC_Data(only_EL=False, drop_duplicate_records=True)
pMHC_data = pMHC.data

human_pMHC_data = pMHC_data[pMHC_data["allele"].str.contains("HLA")]
human_pMHC_data_test = human_pMHC_data.sample(n=100, random_state=428364)
pMHC_data_train = pMHC_data[~pMHC_data.index.isin(human_pMHC_data_test.index)]
pMHC_data_train = pMHC_data_train.reset_index(drop=True)
human_pMHC_data_test = human_pMHC_data_test.reset_index(drop=True)

# Get train and test
print("Getting train and test")
pMHC_data_train = pMHC_data_train[["allele", "peptide", "measurement_inequality", "measurement_value"]]
human_pMHC_data_test = human_pMHC_data_test[["allele", "peptide", "measurement_inequality", "measurement_value"]]

# Get train and es
print("Getting train and es")
train, es = trainf.BalanceSplitData().get_train_val(pMHC_data_train)
train = train.reset_index(drop=True)
es = es.reset_index(drop=True)

# Get pseudoseqs
print("Getting pseudoseqs")
training_allele_pseudoseqs = phf.get_allele_pseudoseqs(train, pMHC)
es_allele_pseudoseqs = phf.get_allele_pseudoseqs(es, pMHC)

training_peptide_pseudoseqs = phf.get_peptide_pseudoseqs(train)
es_peptide_pseudoseqs = phf.get_peptide_pseudoseqs(es)

# BERT_output_dir = "/home/ppaul/mhcglobe/server_test_files/BERT_colab_run_outputs_24_05_01"
# training_allele_pseudoseqs = pd.read_csv(BERT_output_dir + "training_allele_pseudoseqs.csv")
# training_peptide_pseudoseqs = pd.read_csv(BERT_output_dir + "training_peptide_pseudoseqs.csv")
# es_allele_pseudoseqs = pd.read_csv(BERT_output_dir + "es_allele_pseudoseqs.csv")
# es_peptide_pseudoseqs = pd.read_csv(BERT_output_dir + "es_peptide_pseudoseqs.csv")

########################################################################################################

print("Transformers part!!!")
# from transformers import EsmTokenizer, EsmModel
from transformers import EsmTokenizer, TFEsmModel
model_dir = "/scratch/gpfs/ppaul/.cache/huggingface/models--facebook--esm2_t6_8M_UR50D"

tokenizer = EsmTokenizer.from_pretrained(model_dir)
model = TFEsmModel.from_pretrained(model_dir)

# Get BERT embeddings
training_allele_BERT_embeddings = phf.get_BERT_embeddings(training_allele_pseudoseqs, tokenizer, model, "tf")
training_peptide_BERT_embeddings = phf.get_BERT_embeddings(training_peptide_pseudoseqs, tokenizer, model, "tf")
     

es_allele_BERT_embeddings = phf.get_BERT_embeddings(es_allele_pseudoseqs[:5], tokenizer, model, "tf")
es_peptide_BERT_embeddings = phf.get_BERT_embeddings(es_peptide_pseudoseqs[:5], tokenizer, model, "tf")

