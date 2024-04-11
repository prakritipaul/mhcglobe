"""
	Has optimizer and compiler parameters. 
"""

optimizer_params = {1: {"optimizer_type": "RMSprop",
                            "learning_rate": 0.0011339304,
                            "momentum": 0.5,
                            "epsilon": 6.848580326162904e-07,
                            "centered": True},
                    3: {"optimizer_type": "RMSprop",
                            "learning_rate": 0.0019147476,
                            "momentum": 0.5,
                            "epsilon": 3.17051703095139e-07,
                            "centered": True}}

model_compiler_params = {"optimizer": "RMSprop", 
                           "loss": "inequality_loss.MSEWithInequalities().loss",
                           "metrics": ["mean_absolute_error", "mean_squared_error", "root_mean_squared_error"]}

mhcglobe_callbacks = [Callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                mode='min',
                baseline=1,
                min_delta=0.0001)]