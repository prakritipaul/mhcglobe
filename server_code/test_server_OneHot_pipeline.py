import pytest
import pickle
import server_OneHot_MHCGlobe_pipeline_PyTorch_validation as SOneHot

TEST_OUTPUTS_DIR = "/Users/prakritipaul/Library/CloudStorage/GoogleDrive-ppaul@alumni.princeton.edu/My Drive/Pre-industry Professional Prep/AImmunology/server_related/output_files/OneHot_colab_run_outputs_24_05_08/"
PMHC = SOneHot.mhc_data.pMHC_Data(only_EL=False, drop_duplicate_records=True)
TEST_SAMPLE_NUM, TEST_RANDOM_STATE, TEST_MAKEDIR = 100, 428364, False


##### Checking if "get_train_test_data" function works with csv inputs, and w/ w/makedir #####
# and also checking if new dirs are made when they already exist.

pMHC = SOneHot.mhc_data.pMHC_Data(only_EL=False, drop_duplicate_records=True)
# CHANGE in della
test_dir = "/Users/prakritipaul/Library/CloudStorage/GoogleDrive-ppaul@alumni.princeton.edu/My Drive/Pre-industry Professional Prep/AImmunology/server_related/output_files/BERT_colab_run_outputs_24_05_01/"

test_train_data_csv = test_dir + "all_columns_pMHC_data_train_428364.csv"
test_test_data_csv = test_dir + "all_columns_human_pMHC_data_test_428364.csv"

# CHANGE in della
test_out_dir = "/Users/prakritipaul/Library/CloudStorage/GoogleDrive-ppaul@alumni.princeton.edu/My Drive/Pre-industry Professional Prep/AImmunology/server_related/output_files/25_05_07-VS-BERT_colab_run_outputs_24_05_08/"
test_out_train_data_csv = "test_train_25_05_07b.csv"
test_out_test_data_csv = "test_test_25_05_07b.csv"

train_data, test_data = SOneHot.get_train_test_data(pMHC, sample_num=None, random_state=None, train_data_csv=test_train_data_csv, test_data_csv=test_test_data_csv, makedir=False, out_dir=test_out_dir, out_train_data_csv=test_out_train_data_csv, out_test_data_csv=test_out_test_data_csv)


def test_get_train_test_data():
    """
    Test that function returns correct dataframes when given a test set size
    and random state.
    """
    test_train_data, test_data = SOneHot.get_train_test_data(PMHC, sample_num=TEST_SAMPLE_NUM, random_state=TEST_RANDOM_STATE)

    test_train_data = test_train_data[["allele", "peptide", "measurement_inequality", "measurement_value"]]
    test_data = test_data[["allele", "peptide", "measurement_inequality", "measurement_value"]]

    to_open = TEST_OUTPUTS_DIR + "train_test_dataframes.pkl"
    with open(to_open, 'rb') as f:
        train_test_dataframes = pickle.load(f)
    train_data_frame, test_data_frame = train_test_dataframes["training_data"], train_test_dataframes["testing_data"]

    assert test_train_data.equals(train_data_frame)
    assert test_data.equals(test_data_frame)


##### Checking if get_train_es_data works with a given training dataframe #####

test_train_data, test_data = SOneHot.get_train_test_data(PMHC, sample_num=TEST_SAMPLE_NUM, random_state=TEST_RANDOM_STATE, train_data_csv=None, test_data_csv=None, makedir=TEST_MAKEDIR, out_dir=None, out_train_data_csv=None, out_test_data_csv=None)
test_train, test_es = SOneHot.get_train_es_data(train_data_df=test_train_data, train_data_csv=None, es_data_csv=None, makedir=False, out_dir=None, out_train_data2_csv=None, out_es_data_csv=None)
# print(test_train.head(5))
# print(test_es.head(5))

# uploading csvs
test_train_data_csv = TEST_OUTPUTS_DIR + "train_seed-428364_24_05_08.csv"
test_es_data_csv=TEST_OUTPUTS_DIR + "es_seed-428364_24_05_08.csv"
# test_makedir = False
test_makedir = True
test_out_dir = "/Users/prakritipaul/Library/CloudStorage/GoogleDrive-ppaul@alumni.princeton.edu/My Drive/Pre-industry Professional Prep/AImmunology/server_related/output_files/es_train_OneHot/"
test_out_train_data2_csv = "test_out_train_data2b.csv"
test_out_es_data_csv = "test_out_es_datab.csv"

test_train, test_es = SOneHot.get_train_es_data(train_data_df=None, train_data_csv=test_train_data_csv, es_data_csv=test_es_data_csv, makedir=test_makedir, out_dir=test_out_dir, out_train_data2_csv=test_out_train_data2_csv, out_es_data_csv=test_out_es_data_csv)


def test_get_OneHot_features_Y_with_csvs():
    # Input values
    test_train_csv = TEST_OUTPUTS_DIR + "train_seed-428364_24_05_08.csv"
    test_es_csv = TEST_OUTPUTS_DIR + "es_seed-428364_24_05_08.csv"
    test_test_csv = TEST_OUTPUTS_DIR + "all_columns_test_seed-428364_24_05_08.csv"
    test_X_train, test_X_es, test_X_test, test_Y_train, test_Y_es, test_Y_test = SOneHot.get_OneHot_features_Y(train=None, es=None, test=None, train_csv=test_train_csv, es_csv=test_es_csv, test_csv=test_test_csv)

    # True values
    train_es_pickle = TEST_OUTPUTS_DIR + "OneHot_train_es_encodings.pkl"
    with open(train_es_pickle, 'rb') as f:
        OneHot_train_es_encodings = pickle.load(f)
    X_tr, Y_tr, X_es, Y_es = OneHot_train_es_encodings["X_tr"], OneHot_train_es_encodings["Y_tr"], OneHot_train_es_encodings["X_es"], OneHot_train_es_encodings["Y_es"]

    test_pickle = TEST_OUTPUTS_DIR + "OneHot_test_encodings.pkl"
    with open(test_pickle, 'rb') as g:
        OneHot_test_encodings = pickle.load(g)
    X_test, Y_test = OneHot_test_encodings["test_X"], OneHot_test_encodings["test_Y"]

    def check_equality_OneHot_encodings(encoding_1, encoding_2):
        # assuming that the 2 encodings are of the same length
        for i in range(len(encoding_1)):
            if not (encoding_1[0] == encoding_2[0]).all():
                return False
        return True

    assert check_equality_OneHot_encodings(test_X_train, X_tr)
    assert check_equality_OneHot_encodings(test_Y_train, Y_tr)

    assert check_equality_OneHot_encodings(test_X_es, X_es)
    assert check_equality_OneHot_encodings(test_Y_es, Y_es)

    assert check_equality_OneHot_encodings(test_X_test, X_test)
    assert check_equality_OneHot_encodings(test_Y_test, Y_test)


##### get_OneHot_features_Y works when you give it data frames. #####

test_train_data, test_data = SOneHot.get_train_test_data(PMHC, sample_num=TEST_SAMPLE_NUM, random_state=TEST_RANDOM_STATE)
test_train, test_es = SOneHot.get_train_es_data(train_data_df=test_train_data)
# 583650, 145788, 100/ 583650, 145788, 100
X_train, X_es, X_test, Y_train, Y_es, Y_test = SOneHot.get_OneHot_features_Y(train=test_train, es=test_es, test=test_data)

##### recompile model
test_model_3 = SOneHot.get_model(3)
SOneHot.recompile_model(test_model_3)


    

