"""ChessWarrior configuration"""

import os

# -----------------------------------
# configurations about some PARAMETERS
# -----------------------------------

class ResourceConfig(object):
    """resouces configuratioins"""
    cur_dir = os.path.abspath(__file__)

    d = os.path.dirname
    base_dir = d(d(cur_dir))

    _base_data_dir = os.path.join(base_dir, "data")

    best_model_dir = os.path.join(_base_data_dir, "model")

    _sl_base_data_dir = os.path.join(_base_data_dir, "human_expert")

    sl_raw_data_dir = os.path.join(_sl_base_data_dir, "raw") #pgn files

    sl_processed_data_dir = os.path.join(_sl_base_data_dir, "processed") #json files

    rl_model_dir = os.path.join(_base_data_dir, "model/rl")

    json_size = 1024 #moves in a json file

    min_elo = 600.0 #min_elo weight = 0
    max_elo = 2350.0 #max_elo weight = 1

class ModelConfig(object):
    """Model Configuration"""
    cnn_filter_num = 256
    res_layer_num = 39
    cnn_first_filter_size= 5
    cnn_filter_size = 3
    l2_regularizer = 1e-5
    value_fc_size = 512
    drop_out_rate = 0.5
    features = 18


class TrainerConfig(object):
    """Training Configuration"""
    batch_size = 4096
    learning_rate = 0.002
    epoches = 10
    save_interval = 1
    test_interval = 5


class PlayerConfig(object):
    """Playing Configuration"""
    cur_dir = os.path.abspath(__file__)
    d = os.path.dirname
    base_dir = d(d(cur_dir))

    oppo_move_dir = os.path.join(base_dir, "oppo_move.txt")
    ai_move_dir = os.path.join(base_dir, "ai_move.txt")
    

class RLConfig(object):
    lr = 0.001
    gamma = 0.99
    tau = 1.00
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 50
    num_processes = 4
    num_steps = 75 
    max_episode_length = 100 

class Config(object):
    """Configurations"""
    CMD = ['train', 'play', 'data']

    cuda_avaliable = True

    resources = ResourceConfig()

    model = ModelConfig()

    training = TrainerConfig()

    playing = PlayerConfig()

    rl = RLConfig()

