# Explanation of key classes/functions in each of the python scripts pertaining to MHCGlobe.

(1) binding_affinity.py
# convert binding affinity values
def to_ic50
def from_ic50

(2) build_deepnet.py (UNCLEAR)
- class BuildModel():
	"""Create model architecture for MHCGlobe."""
	
	- (function) build_graph(self, hparams):
        """ Build Tensorflow graph for neural network model."""

(3) feature_column_names.py
- (function) get_feature_cols():
    return [
        # feature set 1
        'data_aa_pos_1','data_aa_pos_2','data_aa_pos_3','data_aa_pos_4','data_aa_pos_5',
        'data_aa_pos_6','data_aa_pos_7','data_aa_pos_8','data_aa_pos_9','data_aa_pos_10',
        'data_aa_pos_11','data_aa_pos_12','data_aa_pos_13','data_aa_pos_14','data_aa_pos_15',
        'data_aa_pos_16','data_aa_pos_17','data_aa_pos_18','data_aa_pos_19','data_aa_pos_20',
        'data_aa_pos_21','data_aa_pos_22','data_aa_pos_23','data_aa_pos_24','data_aa_pos_25',
        'data_aa_pos_26','data_aa_pos_27','data_aa_pos_28','data_aa_pos_29','data_aa_pos_30',
        'data_aa_pos_31','data_aa_pos_32','data_aa_pos_33','data_aa_pos_34',
        # feature set 2
        'N1_dist','N2_dist','N3_dist','N4_dist','N5_dist',
        'N6_dist','N7_dist','N8_dist','N9_dist','N10_dist',
        # feature set 3
        'N1_data','N2_data','N3_data','N4_data','N5_data',
        'N6_data','N7_data','N8_data','N9_data','N10_data',
        # feature set 4
        'dist_bin_0.0','dist_bin_0.1','dist_bin_0.2','dist_bin_0.3','dist_bin_0.4',
        'dist_bin_0.5','dist_bin_0.6','dist_bin_0.7',
        # feature set 5
        'data_size'
    ]

(4) inequality_loss.py
(Has various loss metrics) 
- (function) val_loss
- class MSEWithInequalities
	- code derived from MHCFlurry

(5) mhc_data.py

(6) sequence_functions.py
- class PseudoSeq
	- has {allele: pseudo sequence} dicts and dfs

- class MHCName
	- (KEY function) standardize_allele
		Example:
			alleles_vct = ["HLA-A*03:01", "HLA-A*02:01", "HLA-A2402"]
	
        	        allele_dict = {'HLA-A*03:01': 'HLA-A*03:01',
                              'HLA-A*02:01': 'HLA-A*02:01',
                              'HLA-A2402': 'HLA-A*24:02'}
	- Q: unclear when this was used!
	- functions parse allele names to a standard convention ("HLA-A*03:01" -> "HLA-A0301")

- class SeqRepresentation()
	- (function) one_hot: Make One-Hot np array for a residue
		Example:
	            "A" (string) -> array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                   0., 0., 0.]) (np array)
	- (function) select_peptideencoding
		This can be used for both peptides and MHCs.
		Make dictionary for encoding amino acid strings as BLOSUM62 or One Hot.
	        len = 20.

       		If encode_type == "ONE_HOT":
            		{'A': array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                        	     0., 0., 0.]),
             		'R': array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                         	    0., 0., 0.])...}
        
        		# Each value is a list.
        		If encode_type == "BLOSUM62":
            		{'A': [4,
                  		-1,
                  		-2,
                  		-2,
                  		0,
                  		-1,
                  		-1,
                  		0,
                  		-2,
                 		-1,
                  		-1,
                  		-1,
                  		-1,
                  		-2,
                  		-1,
                  		1,
                  		0,
                  		-3,
                  		-2,
                  		0],
            		'R': [-1,
                  		5,
                  		0,
                  		-2,
                  		-3,
                  		1,
                  		0,
                  		-2,
                  		0,
                  		-3,
                  		-2,
                  		2,
                  		-1,
                  		-3,
                  		-2,
                  		-1,
                  		-1,
                  		-3,
                  		-2,
                  		-3]...}

	- (KEY FUNCTION) encode_peptide/mhc
		Make an array of the encoding of a peptide        	
		Note: self.BLOSUM is EITHER BLOSUM or ONE_HOT
		        Example:
		            "ARD" -> array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
		                             0., 0., 0., 0.],
		
                		            [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
		                             0., 0., 0., 0.],

		                            [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
		                             0., 0., 0., 0.]])
- (KEY function) seq_to_15mer
    	Makes input peptide representation in the style of MHCFlurry.
	Example 1: ARDV (4) -> ARDV-X7-ARDV
	Example 2: ARDVA (5) -> ARDV-X7-RDVA
	Example 3: ARDVAA (6) -> ARDV-X7-DVAA
(X7 padding is true until len(peptide) = 8)

	Example 4: ARDVAAAAA (9) -> ARDV-XXX-A-XXX-AAAA
	Example 5: ARDVAAAAAA (10) -> ARDV-XXX-AA-XX-AAAA

- (KEY function) get_XY
    Input: df with columns "allele", "peptide", "measurement_value"
    Output:
        df with characteristics below:

        test_XY = sequence_functions.get_XY(test_df, "ONE_HOT")
            - len = 2 (X, Y) if get_Y=True; else len = 1 (X)
        - test_X_alleles = test_XY[0][0]
            test_X_alleles contain encodings of all allele pseudosequences (len = #alleles= #peptides)
        - test_X_allele_1 = test_X_alleles[0]; 
            test_X_allele_1 is ONE_HOT encoding of the 1st allele's pseudosequence.
            len = 34

        Repeat the same process for MHCFlurry 1.2 peptide encodings, just with:
        - test_X_peptides = test_XY[0][1]
            test_X_peptides contain ONE_HOT encodings of 15-mer MHCFlurry representation
                of all peptides (len = #alleles= #peptides)
        - test_X_peptide_1 = test_X_peptides[0]
            len = 15

(7) paths.py
Literally just the paths to pertinent files.

(8) TO DO train_functions.py

(9) TO DO inequality_loss.py

(10) (SKIPPED) my_functions.py

(*11) mhcglobe.py

				















