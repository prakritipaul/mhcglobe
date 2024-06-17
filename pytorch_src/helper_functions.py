"""
	Helper Functions
"""
import pandas as pd
import numpy as np

class MHCData():
	# data_dir = "/Users/prakritipaul/Git/mhcglobe/data/"
	def __init__(self, mhc_type, data_dir):
		self.data_dir = data_dir
		self.mhc_type = mhc_type

		if self.mhc_type == "I":
			full_training_data_csv = data_dir + "mhcglobe_full_train_data.csv"
			self.full_training_data = pd.read_csv(full_training_data_csv)
			
			allele2seq_csv = data_dir + "allele_sequences_seqlen34.csv"
			self.allele2seq = pd.read_csv(allele2seq_csv)
		
		elif self.mhc_type == "II":
			pass
	
	def get_training_data(self, only_EL, drop_duplicate_records=False):
		pass
	   
test = MHCData(mhc_type="I", data_dir="/Users/prakritipaul/Git/mhcglobe/data/")

class SequenceOperations():
	def __init__(self):
		pass

def get_one_hot_encoding(aa):
	# get a one-hot encoding for an amino acid.
	amino_acids = 'A R N D C Q E G H I L K M F P S T W Y V'.split()
	aa = aa.upper()
	one_hot_encoding = np.zeros(20)
	aa_index = amino_acids.index(aa)
	one_hot_encoding[aa_index] = 1
	return one_hot_encoding

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
