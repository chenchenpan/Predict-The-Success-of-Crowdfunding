#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import time
import joblib
import collections
import argparse
import os
import json
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences


## check the package version
# print(pd.__version__)

# # Preprocess data steps:

# 1. split dataset
# 2. transfer datetime data
# 3. encode categorical data
# 4. encode boolean type data
# 5. normalize data

def main():

    parser = argparse.ArgumentParser()

    # parameters for select input data and metedata configure files
    parser.add_argument('--data_dir', type=str,
        # default='/data/home/t-chepan/projects/MS-intern-project/raw_data',
        help=('directory to load the raw data.'))

    parser.add_argument('--data_name', type=str,
        # default='kickstarter',
        help=('which data will be used? (kickstarter Or indiegogo?)'))

    parser.add_argument('--config_file', type=str,
        # default='config.json',
        help=('which configure file (metadata) will be used?'))

    # parameter for using text features
    parser.add_argument('--use_text_features', type=str2bool, nargs='?',
        const=True, default=False,
        help=('whether encode the text features or not?'))

    parser.add_argument('--encode_text_with', type=str,
        # default='tfidf',
        help=('how to encode the text features? (tfidf, bow, glove)'))

    parser.add_argument('--glove_dir', type=str,
        # default='/data/home/t-chepan/projects/MS-intern-project/raw_data',
        help=('directory to load the pre-trained GloVe.'))

    parser.add_argument('--glove_file', type=str,
        # default='/data/home/t-chepan/projects/MS-intern-project/raw_data',
        help=('which GloVe file will be used? (glove.840B.300d.txt)'))

    parser.add_argument('--max_num_words', type=int,
        # default='/data/home/t-chepan/projects/MS-intern-project/raw_data',
        help=('what is the maximum number of words for encoding text?'))

    parser.add_argument('--max_sequence_length', type=int,
        # default='/data/home/t-chepan/projects/MS-intern-project/raw_data',
        help=('what is the maximum sequence length for encoding text?'))

    parser.add_argument('--embedding_dim', type=int,
        # default='/data/home/t-chepan/projects/MS-intern-project/raw_data',
        help=('what is the embedding_dim of the GloVe?'))


    args = parser.parse_args()


    ### load raw data and related metadata configure file
    if args.data_name == 'kickstarter':
        path_to_data = os.path.join(args.data_dir, "KICK")
    elif args.data_name == 'indiegogo':
        path_to_data = os.path.join(args.data_dir, "INDI")
    else:
        raise argparse.ArgumentTypeError(args.data_name, "can't be recognized.")

    print("Start to load data...")

    train_path = os.path.join(path_to_data, "textandmeta_train.tsv")
    dev_path = os.path.join(path_to_data, "textandmeta_dev.tsv")
    test_path = os.path.join(path_to_data, "textandmeta_test.tsv")
    df_train = pd.read_csv(train_path, sep='\t')
    df_dev = pd.read_csv(dev_path, sep='\t')
    df_test = pd.read_csv(test_path, sep='\t')

    print('*' * 50)
    print('training set size is {}'.format(df_train.shape[0]))
    print('dev set size is {}'.format(df_dev.shape[0]))
    print('test set size is {}'.format(df_test.shape[0]))


    metadata_path = os.path.join(args.data_dir, args.config_file)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)


    print("Processing data...")

    if args.use_text_features:
        mode = args.encode_text_with

        if mode != 'tfidf' and mode != 'bow' and mode != 'word2vec':
            raise argparse.ArgumentTypeError(mode, "can't be recognized.")

        ytrain, Xtrain_struc, Xtrain_text, vectorizer, scaler, tokenizer = encode_dataset(df_train, metadata, vectorizer=None, scaler=None, tokenizer=None, mode=mode)
        ydev, Xdev_meta, Xdev_text, _, _, _ = encode_dataset(df_dev, metadata, vectorizer=vectorizer, scaler=scaler, tokenizer=tokenizer, mode=mode)
        ytest, Xtest_meta, Xtest_text, _, _, _ = encode_dataset(df_test, metadata, vectorizer=vectorizer, scaler=scaler, tokenizer=tokenizer, mode=mode)

        if mode == 'tfidf':

            token_path = os.path.join(path_to_data, 'tfidf_vectorizer.pkl')
            joblib.dump(tokenizer, token_path)

            Xtrain_text_path = os.path.join(path_to_data, 'Xtrain_text_tfidf.npy')
            np.save(Xtrain_text_path, Xtrain_text)
            Xdev_text_path = os.path.join(path_to_data, 'Xdev_text_tfidf.npy')
            np.save(Xdev_text_path, Xdev_text)
            Xtest_text_path = os.path.join(path_to_data, 'Xtest_text_tfidf.npy')
            np.save(Xtest_text_path, Xtest_text)
            print('Saved the encoded text inputs!')

    else:
        ytrain, Xtrain_meta, _, vectorizer, scaler, _ = encode_dataset(df_train, metadata)
        ydev, Xdev_meta, Xdev_text, _, _, _ = encode_dataset(df_dev, metadata, vectorizer=vectorizer, scaler=scaler)
        ytest, Xtest_meta, Xtest_text, _, _, _ = encode_dataset(df_test, metadata, vectorizer=vectorizer, scaler=scaler)


    ### save the results.
    ytrain_path = os.path.join(path_to_data, 'ytrain.npy')
    np.save(ytrain_path, ytrain)
    ydev_path = os.path.join(path_to_data, 'ydev.npy')
    np.save(ydev_path, ydev)
    ytest_path = os.path.join(path_to_data, 'ytest.npy')
    np.save(ytest_path, ytest)
    print('Saved the outputs!')


    Xtrain_meta_path = os.path.join(path_to_data, 'Xtrain_meta.npy')
    np.save(Xtrain_meta_path, Xtrain_meta)
    Xdev_meta_path = os.path.join(path_to_data, 'Xdev_meta.npy')
    np.save(Xdev_meta_path, Xdev_meta)
    Xtest_meta_path = os.path.join(path_to_data, 'Xtest_meta.npy')
    np.save(Xtest_meta_path, Xtest_meta)
    print('Saved the encoded meta inputs!')
    
    ### save all the token 
    vectorizer_path = os.path.join(path_to_data, 'cat_vectorizer.pkl')
    joblib.dump(vectorizer, vectorizer_path)

    scaler_path = os.path.join(path_to_data, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def separate_input_output_cols(df, metadata):
    """According to the metadata, separate the input features, output features and 
        different types of input features.

    Args:
      df: a DataFrame that stores the raw data.
      metadata: a dictionary that stores the detail description for features.
        metadata = {'input_features': ['TenantId','CreatedDate', ...]
                    'output_label': ['AR_exchange_06','AR_sharepoint_06', ...]
                    'input_bool': ['HasEXO','HasSPO', ...],
                    'input_categorical': ['CountryCode', 'Languange', ...],
                    'input_datetime': ['CreatedDate', ...],
                    'input_int': [...] 
                    'input_float': [...]
                    }      
    Returns:
      df_y: a DataFrame that stores the output labels
      df_X_text: a DataFrame that stores the textual input
      df_X_float: a DataFrame that stores the float inputs
      df_X_int: a DataFrame that stores the integer inputs
      df_X_cat: a DataFrame that stores the categorical inputs
      df_X_datetime: a DataFrame that stores the datetime inputs
      df_X_bool: a DataFrame that stores the boolean inputs

    """
    # input_cols = metadata['input_features']
    output_cols = metadata['output_label']
    input_text_cols = metadata['input_text']
    input_float_cols = metadata['input_float']
    input_int_cols = metadata['input_int']
    input_cat_cols = metadata['input_categorical']
    input_datetime_cols = metadata['input_datetime']
    input_bool_cols = metadata['input_bool']

    df_y = df.loc[:, output_cols]
    df_X_text = df.loc[:, input_text_cols]
    df_X_float = df.loc[:, input_float_cols]
    df_X_int = df.loc[:, input_int_cols]
    df_X_cat = df.loc[:, input_cat_cols]
    df_X_datetime = df.loc[:, input_datetime_cols]
    df_X_bool = df.loc[:, input_bool_cols]

    return df_y, df_X_text, df_X_float, df_X_int, df_X_cat, df_X_datetime, df_X_bool


def encode_datetime(df_X_datetime):
    """Encode the datetime inputs from '2/5/2014 5:31:19 AM' format
        to a numerical number of UTC format.

    Args:
      df_: a DataFrame that only stores the datetime inputs.
        
    Returns:
      X_datetime: a numpy array that contains the encoded datetime inputs.
      datetime_cols: a list that contains the datetime colunms name.   
   
    """
    
    cols = df_X_datetime.columns
    for i in cols:
        df_X_datetime[i] = pd.to_datetime(df_X_datetime[i], utc=True,
                            errors='coerce').astype(int,errors='ignore')
        
    X_datetime = df_X_datetime.to_numpy()
    
    return X_datetime


def encode_bool(df_X_bool):
    """Encode the numerical and boolean inputs.
        
    Args:
      df_X_bool: a DataFrame that stores the boolean inputs
        
    Returns:
      X_bool: a numpy array that contains the encoded boolean inputs.

    """
    X_bool = df_X_bool.astype(int).to_numpy()
    return X_bool


def encode_num(df_X_num):
    """Encode the numerical and boolean inputs.
        
    Args:
      df_X_num: a DataFrame that stores the numerical inputs
        
    Returns:
      X_num: a numpy array that contains the float inputs.
      
    """
    X_num = df_X_num.to_numpy()
    return X_num


def encode_strucdata(metadata, df_X_float, df_X_int, df_X_cat, df_X_datetime, df_X_bool, vectorizer, scaler):
    """Encode the meta data part in dataset, such as numerical and categorical data.
        
    """
    print('Starting to encode structural data...')

    # df_y, _, df_X_float, df_X_int, df_X_cat, df_X_datetime, df_X_bool = separate_input_output_cols(df, metadata)
    
    X_list = []
    cols_name = []
    
    if df_X_float.shape[1] > 0:
        X_float = encode_num(df_X_float)
        X_list.append(X_float)
        cols_name += metadata['input_float']

    if df_X_int.shape[1] > 0:
        X_int = encode_num(df_X_int)
        X_list.append(X_int)
        cols_name += metadata['input_int']
    
    if df_X_datetime.shape[1] > 0:
        X_datetime = encode_datetime(df_X_datetime)
        X_list.append(X_datetime)
        cols_name += metadata['input_datetime']
    
    ### normalize all the inputs except boolean, categorical, and text features
    X_arr = np.concatenate(X_list, axis=1)

    if scaler == None:
        scaler = StandardScaler()
        X_struc = scaler.fit_transform(X_arr)
    else:
        X_struc = scaler.transform(X_arr)

    assert len(cols_name) == X_struc.shape[1]
    print('Except boolean, categorical and text input data after encoding, the shape is {}'.format(X_struc.shape))
    print('we have {} columns.'.format(len(cols_name)))

    ### encode boolean columns
    if df_X_bool.shape[1] > 0:
        X_bool = encode_bool(df_X_bool)
        cols_name += metadata['input_bool']
        X_struc = np.concatenate([X_struc, X_bool], axis=1)

    ### encode the categorical columns 
    if df_X_cat.shape[1] > 0:
        X_cat_dict = df_X_cat.to_dict(orient='records')

        if vectorizer == None:   
            vectorizer = DictVectorizer(sparse=False)
            X_cat = vectorizer.fit_transform(X_cat_dict)
            
        else:
            X_cat = vectorizer.transform(X_cat_dict)

        vocab = vectorizer.vocabulary_
        vocab_od = collections.OrderedDict(sorted(vocab.items(), key=lambda x:x[1]))
        cat_encoded_cols = list(vocab_od.keys())
        cols_name += cat_encoded_cols
        X_struc = np.concatenate([X_struc, X_cat], axis=1)

    assert len(cols_name) == X_struc.shape[1]
    print('Non-text input data after encoding, the shape is {}'.format(X_struc.shape))
    print('We have {} columns.'.format(len(cols_name)))
    
    return X_struc, vectorizer, scaler


def open_glove(glove_file_path):

    print('Indexing word vectors.')

    embeddings_index = {}
    f = open(glove_file_path)
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    return embeddings_index



def encode_textdata(df_X_text, tokenizer, mode, max_words, maxlen, embedding_dim, glove_file_path):
    ## encode text columns, encoded text features should not be normalized.

    print('Starting to encode text inputs...')

    texts = df_X_text.iloc[:,0].values.astype('U')
    print('Found %s texts.' % len(texts))


    if mode == 'tfidf':
        # print(df_X_text.values)
        if tokenizer == None:
            tokenizer = Tokenizer(num_words=max_words)
            tokenizer.fit_on_sequences(texts)
        X_text = tokenizer.sequences_to_matrix(texts, mode='tfidf')
        print(X_text.shape)
        embedding_matrix = None

    if mode == 'glove':
        embeddings_index = open_glove(glove_file_path)

        # vectorize the text samples into a 2D integer tensor
        if tokenizer == None:
            tokenizer = Tokenizer(num_words=max_words, oov_token='<UNK>')
            tokenizer.fit_on_texts(texts)
            tokenizer.word_index = {e:i for e,i in tokenizer.word_index.items() if i <= max_words}
            tokenizer.word_index[tokenizer.oov_token] = max_words + 1
        sequences = tokenizer.texts_to_sequences(texts)

        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))

        X_text = pad_sequences(sequences, maxlen=maxlen, padding='post')

        # prepare embedding matrix
        embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
    
    return X_text, embedding_matrix, tokenizer


def encode_dataset(df, metadata, vectorizer=None, scaler=None, tokenizer=None, mode=None, 
    max_words=None, maxlen=None, embedding_dim=None, glove_file_path=None):

    print('Starting to encode inputs...')

    df_y, df_X_text, df_X_float, df_X_int, df_X_cat, df_X_datetime, df_X_bool = separate_input_output_cols(df, metadata)

    y = df_y.to_numpy()

    # check if exist non-text data
    if df_X_float.shape[1] + df_X_int.shape[1] + df_X_cat.shape[1] + df_X_datetime.shape[1] + df_X_bool.shape[1] > 0:
        X_struc, vectorizer, scaler = encode_strucdata(metadata, df_X_float, df_X_int, df_X_cat, df_X_datetime, df_X_bool, vectorizer, scaler)
    else:
        X_struc, vectorizer, scaler = None, None, None

    print("complete encoding part of structural data!")

    if mode == None:  
        X_text, embedding_matrix, tokenizer = None, None, None

    else:
        X_text, embedding_matrix, tokenizer = encode_textdata(df_X_text, tokenizer, mode, max_words, 
        	maxlen, embedding_dim, glove_file_path)

    print("complete encoding part of textual data!") 

    return y, X_struc, X_text, vectorizer, scaler, tokenizer, embedding_matrix


# class EncodedDataset(object):

#     def __init__(self, df, metadata, mode):
#         (self.y, self.X_struc, self.X_text, self.vectorizer,
#          self.scaler, self.tokenizer, self.embedding_matrix) = encode_dataset(df, metadata)



if __name__ == '__main__':
    main()


# Separate config (max_words, maxlen, mode) and state (vectorizer and scaler)
class WordEmbeddingConfig(object):

	def __init__(self, max_words, maxlen, embedding_dim, glove_file_path):
		self.max_words = max_words
		self.maxlen = maxlen
		self.embedding_dim = embedding_dim
		self.glove_file_path = glove_file_path

class TFIDFConfig(object):
	def __init__(self, max_words):
		self.max_words = max_words


# text_config=TFIDFEncodeConfig(100)

# encoder = Encoder(metadata, text_config=text_config)

# X_train, y_train = encoder.fit_transform(df_train)
# X_dev, y_dev = encoder.transform(df_dev)
# X_test, y_test = encoder.transform(df_test)

# X_demo, y_demo = encoder.transform(df_demo)

def contain_nontext_features(metadata):
	n_dtype = len(metadata.keys())

	if n_dtype == 1 and 'input_text' in metadata.keys():
		return False
	else:
		return True

class Encoder(object):

	def __init__(self, metadata, text_config=None):
		self.text_config = text_config
		self.metadata = metadata
		self.has_nontext = contain_nontext_features(metadata)


	def fit_transform(self, df):
		if self.text_config is None:
			y, X_struc, X_text, self.vectorizer, self.scaler, _, _ = encode_dataset(df, self.metadata, mode=None)

		elif isinstance(self.text_config, TFIDFConfig):
			y, X_struc, X_text, self.vectorizer, self.scaler, self.tokenizer, _ = encode_dataset(
				df, self.metadata, mode='tfidf', max_words=self.text_config.max_words)
			
		elif isinstance(self.text_config, WordEmbeddingConfig):
			y, X_struc, X_text, self.vectorizer, self.scaler, self.tokenizer, self.embedding_matrix = encode_dataset(
				df, self.metadata, mode='glove', max_words=self.text_config.max_words, maxlen=self.text_config.maxlen, 
				embedding_dim=self.text_config.embedding_dim, glove_file_path=self.text_config.glove_file_path)

		else:
			raise ValueError('Unknown type of config: {}'.format(type(self.text_config)))

		return y, X_struc, X_text


	def transform(self, df):
		if self.text_config is None:
			y, X_struc, X_text, _, _, _, _ = encode_dataset(df, self.metadata, vectorizer=self.vectorizer, scaler=self.scaler)

		elif isinstance(self.text_config, TFIDFConfig):
			y, X_struc, X_text, _, _, _, _ = encode_dataset(df, self.metadata, vectorizer=self.vectorizer,
			 scaler=self.scaler, tokenizer=self.tokenizer, mode='tfidf', max_words=self.text_config.max_words)
			
		elif isinstance(self.text_config, WordEmbeddingConfig):
			pass
			y, X_struc, X_text, _, _, _, self.embedding_matrix = encode_dataset(df, self.metadata, vectorizer=self.vectorizer, 
				scaler=self.scaler, tokenizer=self.tokenizer, mode='glove', max_words=self.text_config.max_words, maxlen=text_config.maxlen, 
				embedding_dim=self.text_config.embedding_dim, glove_file_path=self.text_config.glove_file_path)

		else:
			raise ValueError('Unknown type of config: {}'.format(type(self.text_config)))

		return y, X_struc, X_text



# indie_dataset = Dataset(df_train, df_dev, df_test, metadata, text_config=text_config)




