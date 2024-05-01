## TO DO Download this into environment + export the neptune key ##
# import neptune
# from neptune.types import File

# NEPTUNE_API_TOKEN = os.getenv("NEPTUNE_API_TOKEN")

# # change
# run = neptune.init_run(project="ppaul/tests",
#     api_token=NEPTUNE_API_TOKEN,
#     tags = ["server test"],
# )

# # In case you have to access a previous run.

# run = neptune.init_run(project="ppaul/MHC-BERT",
#     api_token=NEPTUNE_API_TOKEN,
#     with_id="MHCBER-6",)


# # Neptune
# run["data/all_columns_training"].upload(new_directory + "testrun_train_428364.csv")
# run["data/all_columns_testing"].upload(new_directory + "testrun_test_428364.csv")

# # Neptune
# run["data/train_428364_24_04_11_BERT"].upload(train_csv_name)
# run["data/es_428364_24_04_11_BERT"].upload(es_csv_name)
# # run["data/es_428364_24_04_05_es_1000"].upload(es_csv_name)

# # Neptune
# run["BERT/model"] = "facebook/esm2_t6_8M_UR50D"

# # Neptune
# run["BERT/num_training_samples"] = len(training_allele_pseudoseqs)
# run["BERT/num_es_samples"] = len(es_allele_pseudoseqs)

# # Neptune
# run["model/model_tf_graph"].upload("/content/model.png")

# # Neptune
# run["model/eric_model_dir"] = model_dir

