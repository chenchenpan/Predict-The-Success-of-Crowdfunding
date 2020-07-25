# we are using TensorFlow 2.0 here

import os
import tensorflow as tf
import numpy as np
import keras
# from keras import Model
from keras import optimizers
from keras.layers import Input, Dense, LSTM, Dropout, Flatten, Concatenate
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from encoder import Encoder


def dense_block(input_tensor, model_config):
    x = input_tensor
    for _ in range(model_config.n_layers_dense):
        x = Dense(model_config.hidden_size_dense, activation='relu')(x)
    return x


def lstm_block(input_tensor, text_config, model_config):
# trick: need to load embedding_matrix file to text_config.embedding_matrix   
    embedding_layer = Embedding(input_dim=text_config.embedding_matrix.shape[0],
        output_dim=text_config.embedding_dim, 
        weights=[text_config.embedding_matrix],
        input_length=text_config.maxlen,
        trainable=False 
        )
    x = embedding_layer(input_tensor)
    for i in range(model_config.n_layers_lstm):
        x = LSTM(model_config.hidden_size_lstm, 
            return_sequences=i < (model_config.n_layers_lstm-1)
            )(x)
        x = Dropout(model_config.dropout_rate_lstm)(x)
    return x


def combine_block(tensor1, tensor2, model_config):
    if tensor1 is None and tensor2 is None:
        raise ValueError('Missing all input_tensors.')

    elif tensor1 is None and tensor2 is not None:
        return tensor2

    elif tensor1 is not None and tensor2 is None:
        return tensor1

    else:
        if model_config.combine == 'concate':
            x = Concatenate(axis=-1)([tensor1, tensor2])

        elif model_config.combine == 'attention':
            pass

        else:
            raise ValueError('Unknown type of combining: {}'.format(model_config.combine))

        return x


def output_block(tensor, model_config):
    x = tensor
    if model_config.n_layers_output > 0:
        for _ in range(model_config.n_layers_output):
            x = Dense(model_config.hidden_size_output, activation='relu')(x)

    if model_config.model_type == 'skip_connections':
        x = x + tensor

    if model_config.task_type == 'classification':
        # binary classification task
        if model_config.num_classes <= 2:
            preds = Dense(1, activation='sigmoid')(x)
        # multi-class classification task
        else:
            preds = Dense(model_config.num_classes, activation='softmax')(x)   

    # regression task
    elif model_config.task_type == 'regression':
        preds = Dense(model_config.num_classes)(x)

    else:
        raise ValueError('Unknown type of task: {}'.format(model_config.task_type))

    return preds


def filter_none(contain_none_list):
    new_list = []
    for x in contain_none_list:
        if x is not None:
            new_list.append(x)
    return new_list



class Model(object):
    def __init__(self, text_config, model_config):
        self.text_config = text_config
        self.model_config = model_config               

    def train(self, y_train, X_train_struc, X_train_text, y_dev, X_dev_struc, X_dev_text):
        if X_train_struc is not None:
            n_features = X_train_struc.shape[1]
            input_tensor_struc = Input(shape=(n_features,),
                                       dtype='float32', 
                                       name='structual_data')
            tensor_struc = dense_block(input_tensor_struc, self.model_config)
        else:
            input_tensor_struc = None
            tensor_struc = None

        if X_train_text is None:
            input_tensor_text = None
            tensor_text = None
            
        elif self.text_config.mode == 'glove':
            input_tensor_text = Input(shape=(self.text_config.maxlen,),
                                      dtype='int32', 
                                      name='textual_data')
            tensor_text = lstm_block(input_tensor_text, self.text_config, self.model_config)

        elif self.text_config.mode == 'tfidf':
            input_tensor_text = Input(shape=(self.text_config.max_words,), 
                                      dtype='float32',
                                      name='textual_data')
            tensor_text = dense_block(input_tensor_text, self.model_config)
            
        else:
            raise ValueError('Unknown mode {}!'.format(self.text_config.mode))
            
        input_tensor = combine_block(tensor_struc, tensor_text, self.model_config)

        preds = output_block(input_tensor, self.model_config)

        input_list = filter_none([input_tensor_struc, input_tensor_text])
        # input_list = []
        # for tensor in [input_tensor_struc, input_tensor_text]:
        #     if tensor is not None:
        #         input_list.append(tensor)
        self.model = keras.Model(input_list, preds)
        
        if self.model_config.optimizer == 'adam':
            opt = optimizers.Adam(lr=self.model_config.learning_rate, clipnorm=self.model_config.clipnorm)
        elif self.model_config.optimizer == 'rmsprop':
            opt = optimizers.rmsprop(lr=self.model_config.learning_rate, clipnorm=self.model_config.clipnorm)
        elif self.model_config.optimizer == 'sgd':
            opt = optimizers.sgd(lr=self.model_config.learning_rate, clipnorm=self.model_config.clipnorm)
        else:
            raise ValueError('Unknown optimizer: {}'.format(self.model_config.optimizer))

        if self.model_config.task_type == 'classification' and self.model_config.num_classes <= 2:
            self.model.compile(loss='binary_crossentropy',
                          optimizer=opt,
                          metrics=['acc'])
        elif self.model_config.task_type == 'classification' and self.model_config.num_classes > 2:
            self.model.compile(loss='categorical_crossentropy',
#                                loss='sparse_categorical_crossentropy',
                          optimizer=opt,
                          metrics=['acc'])
        elif self.model_config.task_type == 'regression':
            self.model.compile(loss='mse',
                          optimizer=opt,
                          metrics=['mse'])
        else:
            raise ValueError('Unknown type of task: {}'.format(self.model_config.task_type))

        model_json = self.model.to_json()
        with open(os.path.join(self.model_config.output_dir, 'model.json'), 'w') as f:
            f.write(model_json)

        checkpointer = ModelCheckpoint(
            filepath=os.path.join(self.model_config.output_dir, 'model_weights.hdf5'),
            monitor='val_loss',
            verbose=1, 
            save_best_only=True)

        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0, 
            patience=self.model_config.patience, verbose=1)

        tensorboard = TensorBoard(log_dir=self.model_config.output_dir, update_freq="batch")

        callbacks_list = [early_stopping, checkpointer, tensorboard]
        
        print(self.model.summary())

        X_train_list = filter_none([X_train_struc, X_train_text])
        X_dev_list = filter_none([X_dev_struc, X_dev_text])

        self.hist = self.model.fit(X_train_list, y_train, 
                 validation_data=(X_dev_list, y_dev), 
                 callbacks=callbacks_list,
                 epochs=self.model_config.n_epochs, 
                 batch_size=self.model_config.batch_size,
                 verbose = self.model_config.verbose
                 )
        
        return self.hist

    
    def predict(self, y_test, X_test_struc, X_test_text):
        X_test_list = filter_none([X_test_struc, X_test_text])
        return self.model.predict(X_test_list)

    def evaluate(self, y_test, X_test_struc, X_test_text):
        X_test_list = filter_none([X_test_struc, X_test_text])
        loss, accuracy = self.model.evaluate(X_test_list, y_test, verbose=self.model_config.verbose)
        return loss, accuracy








