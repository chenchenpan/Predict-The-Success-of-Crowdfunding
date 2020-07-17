import unittest
import numpy as np
from encoder import Encoder, Mapping, open_glove
from encoder_test import get_fake_dataset
from modeling import Model


def get_fake_modelconfig(output_path):
    model_config = Mapping()
    model_config.task_type = 'classification' ## 'classification' or 'regression'
    model_config.n_classes = 3 ## number of classes or number of outputs
    model_config.combine = 'concate' ## or 'attention'
    model_config.model_type = 'mlp' ## default is 'mlp', can be 'skip_connections'
    model_config.n_layers_dense = 2
    model_config.hidden_size_dense = 16
    model_config.n_layers_lstm = 2
    model_config.hidden_size_lstm = 32
    model_config.dropout_rate_lstm = 0.0
    model_config.n_layers_output = 2
    model_config.hidden_size_output = 32
    model_config.optimizer = 'adam' ## 'adam', 'sgd', 'rmsprop'
    model_config.learning_rate = 0.001
    model_config.clipnorm = 5.0
    model_config.patience = 20
    model_config.output_dir = output_path
    model_config.n_epochs = 20
    model_config.batch_size = 1
    model_config.verbose = 0
    return model_config

class TestNNModel(unittest.TestCase):
    def test_lstm(self):
        df_train, df_dev, df_test, metadata = get_fake_dataset(with_text_col=True)

        glove_file_path = 'glove.6B.50d.txt'# need be changed to where you store the pre-trained GloVe file.
        
        text_config = Mapping()
        text_config.mode = 'glove'
        text_config.max_words = 20
        text_config.maxlen = 5
        text_config.embedding_dim = 50
        text_config.embeddings_index = open_glove(glove_file_path) # need to change

        encoder = Encoder(metadata, text_config=text_config)
        y_train, X_train_struc, X_train_text = encoder.fit_transform(df_train)
        y_dev, X_dev_struc, X_dev_text = encoder.transform(df_dev)
        y_test, X_test_struc, X_test_text = encoder.transform(df_test)

        text_config.embedding_matrix = encoder.embedding_matrix

        model_config = get_fake_modelconfig('./outputs_test/logs')

        model = Model(text_config, model_config)
        hist = model.train(y_train, X_train_struc, X_train_text, y_train, X_train_struc, X_train_text)
        # y_dev, X_dev_struc, X_dev_text)

        val_acc_true = 1.0
        self.assertTrue(np.isclose(val_acc_true, hist.history['val_acc'][-1]))





if __name__ == '__main__':
    unittest.main()



# ModelConfig = {
#     'task_type': {'value': 'classification', 'type': 'fixed'},
#     'num_outputs': {'value': 2, 'type': 'fixed'},
#     ''
# }


# self.dropout_rate = dropout_rate
#         self.n_lstm_layers = n_lstm_layers
#         self.lstm_hidden_size = lstm_hidden_size
#         self.hidden_size = hidden_size
#         self.n_layers = n_layers
#         self.max_n_epoch = max_n_epoch
#         self.patience = patience
#         self.learning_rate = learning_rate
#         self.useL2 = use_L2regularizer
#         self.batch_size = batch_size
#         self.opt = optimizer

# TextConfig = {
    
# }

# python preprocessing.y --text_config=""


# preprocessed_with_tfidf/text_config.json


# python experiment.py --data_dir=preprocessed_with_glove 




