import argparse
import os
import json
import numpy as np
from encoder import Mapping, Encoder
from modeling import Model


"""
random search for hyperparameters:
1. user provide a search space
2. we generate a set of hyperparameters (as model_config) within this space
3. feed the medel_config to modeling part
4. save the model, result and the model_config
""" 

# search_space = {
#     'hidden_size_dense': ('log_int', [10, 300]),
#     'hidden_size_lstm': ('log_int', [10, 300]),
#     'hidden_size_output': ('log_int', [10, 300]),
#     'learning_rate': ('log_cont', [0.0001, 1]),
#     'n_layers_dense': ('linear_int', [1, 5]),
#     'n_layers_lstm': ('linear_int', [1, 5]),
#     'n_layers_output': ('linear_int', [1, 5]),
#     'dropout_rate': ('linear_cont', [0, 1]),
#     'patience': ('category', [20, 50]),
#     'n_epochs': ('linear_int', [1, 10]),
#     'batch_size': ('category', [64, 128, 256]),
#     'optimizer': ('category', ['adam', 'sgd', 'rmsprop'])
# }

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--encoded_data_dir', type=str,
        # default='/data/home/t-chepan/projects/MS-intern-project/raw_data',
        help=('directory to load the encoded data.'))

    # this is optional 
    parser.add_argument('--data_name', type=str,
        # default='KICK',
        help=('which data will be used? (kickstarter Or indiegogo?)'))

    parser.add_argument('--search_space_filepath', type=str,
        # default='path/to/search_space.json',
        help=('where to load the search space file?'))

    parser.add_argument('--output_dir', type=str,
        # default='path/to/save/outputs',
        help=('directory to save the trained model and related model_config.'))

    parser.add_argument('--task_type', type=str,
        default='classification',
        help=('what is the type of this task? (classification or regression?)'))

    parser.add_argument('--num_classes', type=int,
        # default='classification',
        help=('what is the number of classes (classification) or outputs (regression)?'))

    parser.add_argument('--model_type', type=str,
        default='mlp',
        help=('what type of NN model you want to try? (mlp or skip_connections?)'))

    parser.add_argument('--num_trials', type=int,
        default= 1,
        help=('how many trials you want to run the model?'))


    args = parser.parse_args()

    
    if args.data_name is not None and args.encoded_data_dir is not None:
        path_to_data = os.path.join(args.encoded_data_dir, args.data_name)
        path_to_save = os.path.join(args.output_dir, args.data_name)
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

    elif args.data_name is None and args.encoded_data_dir is not None:
        path_to_data = args.encoded_data_dir
        path_to_save = args.output_dir

    else:
        raise argparse.ArgumentTypeError(args.data_name + ' or ' + args.encoded_data_dir + " can't be recognized.")


    ###########################################
    ## load encoded training set and dev set ##
    ###########################################

    y_train_path = os.path.join(path_to_data, 'y_train.npy')
    if os.path.exists(y_train_path):
        y_train = np.load(y_train_path, mmap_mode='r')
    else:
        raise ValueError('y_train is not found!')

    X_train_struc_path = os.path.join(path_to_data, 'X_train_struc.npy')
    if os.path.exists(X_train_struc_path):
        X_train_struc = np.load(X_train_struc_path, mmap_mode='r')
    else:
        X_train_struc = None

    X_train_text_path = os.path.join(path_to_data, 'X_train_text.npy')
    if os.path.exists(X_train_text_path):
        X_train_text = np.load(X_train_text_path, mmap_mode='r')
    else:
        X_train_text = None

    y_dev_path = os.path.join(path_to_data, 'y_dev.npy')
    if os.path.exists(y_dev_path):
        y_dev = np.load(y_dev_path, mmap_mode='r')
    else:
        raise ValueError('y_dev is not found!')

    X_dev_struc_path = os.path.join(path_to_data, 'X_dev_struc.npy')
    if os.path.exists(X_dev_struc_path):
        X_dev_struc = np.load(X_dev_struc_path, mmap_mode='r')
    else:
        X_dev_struc = None

    X_dev_text_path = os.path.join(path_to_data, 'X_dev_text.npy')
    if os.path.exists(X_dev_text_path):
        X_dev_text = np.load(X_dev_text_path, mmap_mode='r')
    else:
        X_dev_text = None

    text_config_path = os.path.join(path_to_data, 'text_config.json')
    if os.path.exists(text_config_path):
        with open(text_config_path, 'r') as f:
            text_config = json.load(f)
        text_config = Mapping(text_config)
    else:
        text_config = None

    if text_config is not None and text_config.mode == 'glove':
        embedding_matrix_path = text_config.embedding_matrix_path
        if os.path.exists(embedding_matrix_path):
            embedding_matrix = np.load(embedding_matrix_path, mmap_mode='r')
            text_config.embedding_matrix = embedding_matrix
        else:
            raise ValueError('embedding_matrix is not found!')
    else:
        embedding_matrix = None


    ###########################################
    ## sample model config from search space ##
    ###########################################

    if args.task_type is not None and args.num_classes is not None:
        print('you are choosing ' + args.model_type + ' as the model type!')
        default_model_config = create_default_modelconfig(args.task_type, args.num_classes, args.model_type, path_to_save)
    else:
        raise ValueError('You are missing task_type or num_classes or both!')

    ## load search space file which is provided by users ##
    with open(args.search_space_filepath, 'r') as f:
        search_space = json.load(f)
    search_space = Mapping(search_space)

    
    #######################################################################
    ## update default model_config based on search_space and train model ##
    #######################################################################
  
    for i in range(args.num_trials):
        model_config = sample_modelconfig(search_space, default_model_config)
        model_name = 'model_{}'.format(i)
        print('*' * 20)
        print('model_config: ' + model_config['output_dir'])

        model_config = Mapping(model_config)

        print('*' * 20)
        print('model_config: ' + model_config.output_dir)

        model_config.output_dir = os.path.join(default_model_config.output_dir, model_name)
        if not os.path.exists(model_config.output_dir):
            os.makedirs(model_config.output_dir)
        model = Model(text_config, model_config)
        hist = model.train(y_train, X_train_struc, X_train_text, y_train, X_train_struc, X_train_text)

        ## save hist.history and model_config ##
        history_path = os.path.join(model_config.output_dir, 'history.json')
        with open(history_path, 'w') as hf:
            json.dump(hist.history, hf)

        model_config_savepath = os.path.join(model_config.output_dir, 'model_config.json')
        with open(model_config_savepath, 'w') as mf:
            json.dump(model_config, mf)



def sample_modelconfig(search_space, default_model_config):
    model_config = default_model_config.copy()
    for k, v in search_space.items():
        if v[0] == 'linear_int':
            model_config[k] = np.random.randint(v[1][0], v[1][1])
            
        if v[0] == 'linear_cont':
            model_config[k] = np.round(np.random.uniform(v[1][0], v[1][1]), 2)
            
        if v[0] == 'log_cont':
            model_config[k] = np.random.uniform(np.log(v[1][0]), np.log(v[1][1]))
            model_config[k] = np.round(np.exp(model_config[k]), 5)
        
        if v[0] == 'log_int':
            model_config[k] = np.random.uniform(np.log(v[1][0]), np.log(v[1][1]))
            model_config[k] = int(np.round(np.exp(model_config[k])))
            
        if v[0] == 'category':
            model_config[k] = v[1][np.random.randint(len(v[1]))]

    return model_config
    


def create_default_modelconfig(task_type, num_classes, model_type, output_dir):
    model_config = Mapping()
    model_config.task_type = task_type ## 'classification' or 'regression'
    model_config.num_classes = num_classes ## number of classes or number of outputs
    model_config.combine = 'concate' ## or 'attention'
    model_config.model_type = model_type ## default is 'mlp', can be 'skip_connections'
    model_config.n_layers_dense = 2
    model_config.hidden_size_dense = 16
    model_config.n_layers_lstm = 2
    model_config.hidden_size_lstm = 32
    model_config.dropout_rate_lstm = 0.0
    model_config.n_layers_output = 2
    model_config.hidden_size_output = 32
    model_config.optimizer = 'adam' 
    model_config.learning_rate = 0.001
    model_config.clipnorm = 5.0
    model_config.patience = 20
    model_config.output_dir = output_dir
    model_config.n_epochs = 20
    model_config.batch_size = 1
    model_config.verbose = 0
    return model_config 


if __name__ == '__main__':
    main()
