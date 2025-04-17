from percepformer import config, Trainer
import optuna
import hashlib
import json


def unique_name_from_dict(d):
    dict_str = json.dumps(d, sort_keys=True)  # Convert dict to sorted JSON string
    return hashlib.sha256(dict_str.encode()).hexdigest()[:10]  # Shorten for readability


# create objective for optuna hyper parameter searching


def objective(trial, cfg):
    # create global configuration options for all trials
    cfg["train"]["scheduler"]["parameters"]["step_size"] = 100
    cfg["train"]["epochs"] = 60
    cfg["train"]["log_interval"] = 15  # dose not display any thing
    cfg["train"]["loss_weight_fcn"] = "x*x+1"  # is number 1(x=0) to 2(x=1)
    cfg["data"]["pkl_pathes"] = (
        [  # Paths to your dataset pickle files (ensure they exist)
            "./DATA/pip/EURUSD-1h.pkl",  # 1-hour data
            "./DATA/pip/EURUSD-30m.pkl",  # 30-minute data
            "./DATA/pip/EURUSD-15m.pkl",  # 15-minute data
        ]
    )

    cfg["train"]["batch_size"] = trial.suggest_categorical("batch_size", [64, 128])
    cfg["optimizer"]["parameters"]["lr"] = trial.suggest_float(
        "lr", 1e-5, 1e-2, log=True
    )
    cfg["model"]["parameters"]["d_model"] = trial.suggest_categorical(
        "d_model", [32, 64, 128]
    )
    cfg["model"]["parameters"]["num_blocks"] = trial.suggest_int("num_blocks", 2, 5)
    cfg["model"]["parameters"]["act_fun"] = trial.suggest_categorical(
        "act_fun", ["tanh", "relu", "gelu"]
    )
    cfg["model"]["parameters"]["num_encoder_layers"] = trial.suggest_int(
        "num_blocks", 3, 10
    )

    # create unique name for trining data
    uname = unique_name_from_dict(cfg)
    cfg["train"]["checkpoint_dir"] = f"./DATA/checkpoints/train_{uname}"

    # Train & validate model
    trainer = Trainer(cfg)
    minimum_valid_loss = trainer.train()

    return minimum_valid_loss


# hyper searching
def objfun(trial):
    return objective(trial, config)


study = optuna.create_study(direction="minimize")
study.optimize(objfun, n_trials=100, n_jobs=4)  # Run 4 processes in parallel
print("Best parameters:", study.best_params)
