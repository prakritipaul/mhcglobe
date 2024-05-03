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

# # Neptune
# BERT_architecture = {"embedding_layer": "last",
# 										 "layers": "2 Dense- separate dim reduction",
# 										 "kernel_regularizer": "l1(l=0.01)",
# 										 "activation": "relu",
# 										 "inputs": "BERT_allele (30, 320); peptide_allele (15, 320)",
# 										 "output": "model on dim reduction of BERT features = 20 dims"}

# run["BERT/BERT_architecture"] = BERT_architecture

"""
### Compilation related
"""

# Neptune
# model 1
# model_optimizer_params = {"learning_rate": 0.0011339304,
#                             "momentum": 0.5,
#                             "epsilon": 6.848580326162904e-07,
#                             "centered": True,
#                             "optimizer_type": "RMSprop"}

# model_3_optimizer_params = {"optimizer_type": "RMSprop",
# 														"learning_rate": 0.0019147476,
# 														"momentum": 0.5,
# 														"epsilon": 3.17051703095139e-07,
# 														"centered": True}

# # Neptune
# model_compiler_params = {"optimizer": "RMSprop",
# 													 "loss": "inequality_loss.MSEWithInequalities().loss",
# 													 "metrics": ["mean_absolute_error", "mean_squared_error", "root_mean_squared_error"]}

# # Converts list into correct format.
# model_compiler_params["metrics"] = json.dumps(model_compiler_params["metrics"])
# run["model/compiler_params"] = model_compiler_params

# Neptune
model_compiler_params = {"optimizer": "RMSprop",
													 "loss": "inequality_loss.MSEWithInequalities().loss",
													 "metrics": ["mean_absolute_error", "mean_squared_error", "root_mean_squared_error"]}

# Converts list into correct format.
model_compiler_params["metrics"] = json.dumps(model_compiler_params["metrics"])
run["model/compiler_params"] = model_compiler_params

# # Neptune
# from neptune.types import File
# run["model/tf_graph"].upload("/content/model.png")


# # Neptune
# training_specs = {"batch_size": 100,
# 									"epochs": 300,
# 									"shuffle": True,
# 									"verbose": 1,
# 									"mhc_callbacks": ["monitor=val_loss", "patience=20", "mode=min", "baseline=1", "min_delta=0.0001"],
# 									"early_stopping": True,
# 									"num_training_samples": 1000,
# 									"num_es_samples": 500}

# # Converts list into correct format.
# training_specs["mhc_callbacks"] = json.dumps(training_specs["mhc_callbacks"])

# # Neptune
# run["training/training_specs"] = training_specs

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
# 		run["training/loss"].append(loss[i])
# 		run["training/val_loss"].append(val_loss[i])
# 		run["training/mae"].append(mae[i])
# 		run["training/val_mae"].append(val_mae[i])
# 		run["training/mse"].append(mse[i])
# 		run["training/val_mse"].append(val_mse[i])
# 		run["training/rmse"].append(rmse[i])
# 		run["training/val_rmse"].append(val_rmse[i])

# """
# ### Save the predictions
# """
# run["testing/test_predictions_428364_24_04_11_BERT"].upload(File.as_pickle(test_predictions))
# run["testing/test_predictions_df_all_24_04_11_BERT"].upload(File.as_pickle(prediction_df_all))
# run["testing/test_predictions_df_all_viz_428364_24_04_11_BERT"].upload(File.as_html(prediction_df_all))

# # Neptune
# run["testing/r_squared"] = predictions_r
# run["testing/mse"] = predictions_mse
# run["testing/scatter_plot"].upload((savefig_name))

# # end neptune
# run.stop()