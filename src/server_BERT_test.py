import numpy as np
import pandas as pd
import joblib as jb
import json

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.callbacks as Callbacks
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers, losses

import os
import sys
sys.path.append("/content/mhcglobe/src")
import mhcglobe
import mhc_data
import inequality_loss
import train_functions as trainf
import binding_affinity as ba
import sequence_functions as seqf
import prakriti_helper_functions as phf

# Get data
pMHC = mhc_data.pMHC_Data(only_EL=False, drop_duplicate_records=True)
pMHC_data = pMHC.data

human_pMHC_data = pMHC_data[pMHC_data["allele"].str.contains("HLA")]
human_pMHC_data_test = human_pMHC_data.sample(n=100, random_state=428364)
pMHC_data_train = pMHC_data[~pMHC_data.index.isin(human_pMHC_data_test.index)]
pMHC_data_train = pMHC_data_train.reset_index(drop=True)
human_pMHC_data_test = human_pMHC_data_test.reset_index(drop=True)

# Get train and test
pMHC_data_train = pMHC_data_train[["allele", "peptide", "measurement_inequality", "measurement_value"]]
human_pMHC_data_test = human_pMHC_data_test[["allele", "peptide", "measurement_inequality", "measurement_value"]]

# Get train and es
train, es = trainf.BalanceSplitData().get_train_val(pMHC_data_train)
train = train.reset_index(drop=True)
es = es.reset_index(drop=True)

# Get pseudoseqs
training_allele_pseudoseqs = phf.get_allele_pseudoseqs(train, pMHC)
es_allele_pseudoseqs = phf.get_allele_pseudoseqs(es, pMHC)

training_peptide_pseudoseqs = phf.get_peptide_pseudoseqs(train)
es_peptide_pseudoseqs = phf.get_peptide_pseudoseqs(es)

########################################################################################################

# from transformers import EsmTokenizer, EsmModel
from transformers import EsmTokenizer, TFEsmModel
model_dir = "/scratch/gpfs/ppaul/.cache/huggingface/models--facebook--esm2_t6_8M_UR50D"

tokenizer = EsmTokenizer.from_pretrained(model_dir)
model = TFEsmModel.from_pretrained(model_dir)

# Get BERT embeddings
training_allele_BERT_embeddings = phf.get_BERT_embeddings(training_allele_pseudoseqs, tokenizer, model, "tf")
training_peptide_BERT_embeddings = phf.get_BERT_embeddings(training_peptide_pseudoseqs, tokenizer, model, "tf")
     