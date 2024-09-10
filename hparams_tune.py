from sklearn.model_selection import ParameterGrid
from train_lstm import train_lstm
import pandas as pd
import os

if __name__ == "__main__":

    main_dir = "C:\\UofL - MSI\\DARPA"

    main_config = {}
    #main_config["root_dir"] = "C:\\UofL - MSI\\DARPA\\mat_data\\user_wise"
    #main_config["save_dir"] = "C:\\UofL - MSI\\DARPA\\codes\\experiments\\EXP_LSTM_1"    
    main_config["SEED"] = 123
    main_config["VERBOSE"] = False
    main_config["RESUME"] = False
    main_config["IN_CH"] = 2
    main_config["OUTPUT_SIZE"] = 5
    main_config["EPOCHS"] = 100
    main_config["ES_PATIENCE"] = 15
    main_config["PRINT_EVERY"] = 10
    main_config["BS"] = 1024

    grid = {}
    grid["HIDDEN_SIZE"] = [ 2**i for i in range(5, 11)] # 2**5 to 1024
    grid["NUM_LAYERS"] = [1, 2, 4, 8]
    grid["LR"] = [1e-3, 1e-4, 1e-5]
    grid["DROPOUT"] = [0.0, 0.3]
    grid["t"] = [2,4,6,8]
    grid["MODEL"] = ["LSTM", "CNN", "MLP"]

    print(grid)
    
    result_dict = {}
    result_dict['acc'] = []
    result_dict['f1'] = []
    for key in grid.keys():
        result_dict[key] = []
    
    for i, g in enumerate(ParameterGrid(grid)):
        t = g['t']
        # main_config["root_dir"] = f"C:\\UofL - MSI\\DARPA\\mat_data\\user_wise_{t}s"
        main_config["root_dir"] = os.path.join(main_dir, "mat_data", f"user_wise_{t}s")
        main_config["save_dir"] = os.path.join(main_dir, "experiments", f"EXP_LSTM_{i:04d}_{t}_{g['MODEL']}s")
        #main_config["save_dir"] = f"C:\\UofL - MSI\\DARPA\\experiments\\EXP_LSTM_{i:04d}_{t}_{g['MODEL']}s"
        main_config["LENGTH"] = 64 * t
        config = {**main_config, **g} # add g to main_config
        acc, f1 = train_lstm(config)
        #acc, f1 = [0.0, 0.0]
        for key in g.keys():
            result_dict[key].append(g[key])
        result_dict['acc'].append(acc)
        result_dict['f1'].append(f1)
        print(f"Exp {i} - {g} - Acc: {acc} - F1: {f1}")
        # if i == 10: 
        #     break
    
    df = pd.DataFrame(result_dict)
    df.to_csv("hparams_tune.csv")