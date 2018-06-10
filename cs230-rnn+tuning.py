
# coding: utf-8

# In[1]:


#%matplotlib inline
import numpy as np
import pandas as pd
import sklearn
import matplotlib as plt
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle 
import matplotlib.pyplot as plt
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model, datasets
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import naive_bayes


# In[2]:


df = pd.read_csv('train_shuffled.csv')
# df = pd.read_csv('train_balanced.csv')
# df = shuffle(df)
# df.to_csv('train_shuffled.csv')


# In[3]:


# count the number of successful projects and failed projects
succ_num = sum(df['final_status'] == 1)
print(succ_num)
total_num = df.shape[0]
print(total_num)
print(1- succ_num/total_num)


# In[4]:


# df.insert(13, 'duration', df['deadline']-df['launched_at'])
# df.head()


# # Data preprocessing
# 1. balance the dataset (optional)
# 3. encode category features
# 4. encode text features
# 4. modify the dataset: add 'duration', drop colunms as 'name', 'project_id', etc.
# 3. split dataset to training, dev and test set (90%, 5%, 5%)

# In[5]:


# Make the dataset balance: half successful projects and half failed projects
def balance(df):
    # seperate successful projects and failed projects
    df_succ = df[df['final_status'] == 1]
    df_fail = df[df['final_status'] == 0]
    # duplicate successful projects
    df_succ_copy = df_succ.copy()
    # random select failed projects and its amount equals to 2 times of sucessful projects
    df_fail_sel = df_fail.sample(n = succ_num*2)
    # concat the 3 dataframes
    df_balance = pd.concat([df_succ, df_succ_copy, df_fail_sel], axis=0)
    # shuffle the concated dataframe
    df_balance = shuffle(df_balance)
    return df_balance


# In[6]:


# Encode 'category' features, label them with values between 0 and n_classes-1
def encoder_cat(df, col):
    le = preprocessing.LabelEncoder()
    col_label = le.fit_transform(df[col])
    df[col]=pd.Series(col_label)
    return le


# In[7]:


# encode text features
def encoder_text(df, col, min_df=10):
    df[col] = df[col].astype(str)
    vectorizer = CountVectorizer(min_df=min_df)
    vectorizer.fit(df[col])
    col_bag_of_words = vectorizer.transform(df[col])
    return col_bag_of_words, vectorizer


# In[8]:


def modify(df):
    # add a new colunm ‘duration’
    df.insert(13, 'duration', df['deadline']-df['launched_at'])
    # df['duration'] = df['deadline'] - df['launched_at'] 
    # drop unused colunms
    df = df.drop(columns=['Unnamed: 0', 'project_id', 'name', 'desc', 'keywords', 'deadline', 'state_changed_at', 'created_at', 'launched_at', #'backers_count', 
                          'final_status'])
    encoder_cat(df, 'country')
    encoder_cat(df, 'currency')
    encoder_cat(df, 'disable_communication')
    return df


# In[9]:


def data_preprocess(df):
    df_data = modify(df)
    df_keywords_encoded, kw_vec = encoder_text(df, 'keywords', min_df=5)
    df_desc_encoded, desc_vec = encoder_text(df, 'desc', min_df=10)
    df_labels = df['final_status']
    return df_data, df_keywords_encoded, df_desc_encoded, df_labels, {'vectorizer': [kw_vec, desc_vec]}


# In[10]:


# split dataset to training, dev and test
def data_split(df):
    
    df_data, df_keywords_encoded, df_desc_encoded, df_labels, preprocess_info = data_preprocess(df)
    
    n = float(len(df_data))
    n_train = int(n * 0.9)
    n_dev = int(n * 0.05)

    training_data = df_data[:n_train] 
    training_kw = df_keywords_encoded[:n_train]
    training_desc = df_desc_encoded[:n_train]
    training_Y = df_labels[:n_train]

    dev_data = df_data[n_train : (n_train + n_dev)]
    dev_kw = df_keywords_encoded[n_train : (n_train + n_dev)]
    dev_desc = df_desc_encoded[n_train : (n_train + n_dev)]
    dev_Y = df_labels[n_train : (n_train + n_dev)]

    test_data = df_data[(n_train + n_dev) :] 
    test_kw = df_keywords_encoded[(n_train + n_dev) :]
    test_desc = df_desc_encoded[(n_train + n_dev) :]
    test_Y = df_labels[(n_train + n_dev) :]

    # concatinate inputs to ONE single input X
    from scipy.sparse import hstack
    training_X = hstack([#training_data, 
                         training_kw, training_desc])
    dev_X = hstack([#dev_data, 
                    dev_kw, dev_desc])
    test_X = hstack([#test_data, 
                     test_kw, test_desc])
#     training_X = hstack([training_kw, training_desc])
#     dev_X = hstack([dev_kw, dev_desc])
#     test_X = hstack([test_kw, test_desc])
    
    info = {}
    info['data'] = [training_data, dev_data, test_data]
    info.update(preprocess_info)
    
    return training_X, training_Y, dev_X, dev_Y, test_X, test_Y, info



# In[12]:

training_X, training_Y, dev_X, dev_Y, test_X, test_Y, info = data_split(df)
training_data, dev_data, test_data = info['data']

n_train = training_data.shape[0]
n_dev = dev_data.shape[0]
n_test = test_data.shape[0]

labels = df['final_status']

train_label = labels[:n_train]
dev_label = labels[n_train:n_train+n_dev]
test_label = labels[n_train+n_dev:n_train+n_dev+n_test]

scaler = sklearn.preprocessing.StandardScaler().fit(training_data)
norm_train_data = scaler.transform(training_data)
norm_dev_data = scaler.transform(dev_data)
norm_test_data = scaler.transform(test_data)


# # Neural Network 

# In[75]:


import tensorflow as tf
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, LSTM, GlobalMaxPooling1D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Merge, Concatenate
from keras.layers import regularizers
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from sklearn.metrics import confusion_matrix


# In[14]:


# prepare 'desc' for embedding

desc = df['desc'].astype(str)
MAX_DESC_WORDS = 5000

desc_tok = Tokenizer(num_words=MAX_DESC_WORDS, oov_token='<UNK>')
desc_tok.fit_on_texts(desc)

desc_tok.word_index = {e:i for e,i in desc_tok.word_index.items() if i <= MAX_DESC_WORDS}
desc_tok.word_index[desc_tok.oov_token] = MAX_DESC_WORDS + 1

desc_seq = desc_tok.texts_to_sequences(desc)

DESC_MAXLEN = 30
padded_desc = pad_sequences(desc_seq, maxlen=DESC_MAXLEN, padding='post')


# In[15]:


# prepare 'keywords' for embedding

kw = df['keywords'].astype(str)
MAX_KW_WORDS = 5000

kw_tok = Tokenizer(num_words=MAX_KW_WORDS, oov_token='<UNK>')
kw_tok.fit_on_texts(kw)

kw_tok.word_index = {e:i for e,i in kw_tok.word_index.items() if i <= MAX_KW_WORDS}
kw_tok.word_index[kw_tok.oov_token] = MAX_KW_WORDS + 1

kw_seq = kw_tok.texts_to_sequences(kw)

KW_MAXLEN = 10
padded_kw = pad_sequences(kw_seq, maxlen=KW_MAXLEN, padding='post')


# In[16]:


train_desc = padded_desc[:n_train]
dev_desc = padded_desc[n_train:n_train+n_dev]
test_desc = padded_desc[n_train+n_dev:n_train+n_dev+n_test]

train_kw = padded_kw[:n_train]
dev_kw = padded_kw[n_train:n_train+n_dev]
test_kw = padded_kw[n_train+n_dev:n_train+n_dev+n_test]


# In[17]:


# Embedding desc and kw using pre-trained GloVe word embeddings

embeddings_index = {}
f = open('glove.840B.300d.txt')
for line in f:
    values = line.split()
    word = values[0]
    try:
        coefs = np.asarray(values[1:], dtype='float32')
    except:
        print(values[:5])
    embeddings_index[word] = coefs
f.close()


# In[18]:


embedding_matrix_desc = np.zeros((len(desc_tok.word_index) + 1, 300))
for word, i in desc_tok.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix_desc[i] = embedding_vector


# In[19]:


embedding_matrix_kw = np.zeros((len(kw_tok.word_index) + 1, 300))
for word, i in kw_tok.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix_kw[i] = embedding_vector


# # RNN Model

# In[83]:


def train_model(hyp_params, max_n_epoch=50, patience=5, name='model'):
    n_features = 6 if hyp_params['use_backer'] else 5
    
    desc_input = Input(shape=(DESC_MAXLEN,), dtype='int32')
    embedding_layer_desc = Embedding(len(desc_tok.word_index) + 1,
                                300,
                                weights=[embedding_matrix_desc],
                                input_length=DESC_MAXLEN,
                                trainable=False)
    embedded_desc = embedding_layer_desc(desc_input)
    
    x1 = Dropout(hyp_params['dropout_rate'])(embedded_desc)
    
    for i in range(hyp_params['n_lstm_layers']):
        x1 = LSTM(hyp_params['lstm_hidden_size'], return_sequences=i < (hyp_params['n_lstm_layers']-1))(x1)
        x1 = Dropout(hyp_params['dropout_rate'])(x1)


    kw_input = Input(shape=(KW_MAXLEN,), dtype='int32')
    embedding_layer_kw = Embedding(len(kw_tok.word_index) + 1,
                                300,
                                weights=[embedding_matrix_kw],
                                input_length=KW_MAXLEN,
                                trainable=False)
    embedded_kw = embedding_layer_kw(kw_input)

    x2 = Dropout(hyp_params['dropout_rate'])(embedded_kw)
    
    for i in range(hyp_params['n_lstm_layers']):
        x2 = LSTM(hyp_params['lstm_hidden_size'], return_sequences=i < (hyp_params['n_lstm_layers']-1))(x2)
        x2 = Dropout(hyp_params['dropout_rate'])(x2)

    data_input = Input(shape=(n_features,), dtype='float32')
    x3 = Dense(hyp_params['data_fc_hidden_size'])(data_input)

    x = Concatenate(axis=-1)([x1, x2, x3])
    
    for _ in range(hyp_params['n_fc_layers']):
        x = Dense(hyp_params['fc_hidden_size'], activation='relu')(x)

    preds = Dense(1, activation='sigmoid')(x)

    model = Model([desc_input, kw_input, data_input], preds)

    if hyp_params['opt'] == 'adam':
        opt = optimizers.Adam(lr=hyp_params['lr'], clipnorm=hyp_params['clipnorm'])
    elif hyp_params['opt'] == 'rmsprop':
        opt = optimizers.rmsprop(lr=hyp_params['lr'], clipnorm=hyp_params['clipnorm'])
    elif hyp_params['opt'] == 'sgd':
        opt = optimizers.sgd(lr=hyp_params['lr'], clipnorm=hyp_params['clipnorm'])
    else:
        raise ValueError('Unknown optimizer: {}'.format(hyp_params['opt']))

    
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['acc'])
    
    model_json = model.to_json()
    with open("/home/ubuntu/Notebooks/{}.json".format(name), "w") as json_file:
        json_file.write(model_json)
    checkpointer = ModelCheckpoint(filepath='/home/ubuntu/Notebooks/{}_weights.hdf5'.format(name), 
                                   monitor='val_acc',
                                   verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=patience, verbose=1)
    callbacks_list = [early_stopping, checkpointer]

    hist = model.fit([train_desc, train_kw, norm_train_data[:, :n_features]], train_label, 
                     validation_data=([dev_desc, dev_kw, norm_dev_data[:, :n_features]], dev_label), 
                     callbacks=callbacks_list,
                     epochs=max_n_epoch, batch_size=128)

    return hist, model


# In[77]:


# hyper_params = {'dropout_rate': 0.5, 
#      'n_lstm_layers': 1, 'lstm_hidden_size': 32, 
#      'n_fc_layers': 1, 'fc_hidden_size': 32,
#      'use_backer': False,
#      'data_fc_hidden_size': 32,
#      'opt': 'adam', 'lr': 0.001, 'clipnorm': 5.0}
training_data.shape


# In[1]:


# # test for draw confusion metric

# experiments = []
# hp = {'clipnorm': 5.0,
#      'data_fc_hidden_size': 8,
#      'dropout_rate': 0.0,
#      'fc_hidden_size': 8,
#      'lr': 0.001,
#      'lstm_hidden_size': 8,
#      'n_fc_layers': 1,
#      'n_lstm_layers': 1,
#      'opt': 'adam',
#      'use_backer': False}
# print(hp)
# hist, model = train_model(hp, name='love')
# experiments.append({'hyperparam': hp, 'history': hist.history, 
#                     'best_val_loss': min(hist.history['val_loss']), 
#                     'best_val_acc': max(hist.history['val_acc'])})
# print(hist.history['val_acc'])

# # with open('/home/ubuntu/Notebooks/rnn_experiments_nb.json', 'w') as f:
# #     json.dump(experiments, f)


# In[82]:


experiments = []
hp = {'clipnorm': 5.0,
     'data_fc_hidden_size': 64,
     'dropout_rate': 0.48576812493824445,
     'fc_hidden_size': 64,
     'lr': 0.0001457152018119632,
      # 'lr': 0.001,
     'lstm_hidden_size': 64,
     'n_fc_layers': 4,
     'n_lstm_layers': 4,
     'opt': 'adam',
     'use_backer': False}
print(hp)
hist, model = train_model(hp, name='raccoon')
experiments.append({'hyperparam': hp, 'history': hist.history, 
                    'best_val_loss': min(hist.history['val_loss']), 
                    'best_val_acc': max(hist.history['val_acc'])})
print(hist.history['val_acc'])

with open('/home/ubuntu/Notebooks/rnn_experiments_2.json', 'w') as f:
    json.dump(experiments, f)


# In[154]:


for i in range(60):
    hp = {'dropout_rate': 0.5, 
          'n_lstm_layers': 1, 'lstm_hidden_size': 32, 
          'n_fc_layers': 1, 'fc_hidden_size': 32,
          'use_backer': False,
          'data_fc_hidden_size': 32,
          'opt': 'adam', 'lr': 0.001, 'clipnorm': 5.0}

    hp['dropout_rate'] = np.random.uniform(0.0, 0.5)
    hp['n_lstm_layers'] = np.random.randint(2, 5)
    hp['lstm_hidden_size'] = [64, 128, 256][np.random.randint(0, 3)]
    hp['n_fc_layers'] = np.random.randint(2, 5)
    hp['fc_hidden_size'] = [64, 128, 256][np.random.randint(0, 3)]
    hp['data_fc_hidden_size'] = [64, 128, 256][np.random.randint(0, 3)]
    hp['lr'] = 10 ** (np.random.uniform(-2, -4))
    print(hp)
    hist, model = train_model(hp, name="model_{}".format(i))
    experiments.append({'hyperparam': hp, 'history': hist.history, 
                        'best_val_loss': min(hist.history['val_loss']), 
                        'best_val_acc': max(hist.history['val_acc'])})
    print(hist.history['val_acc'])

    with open('/home/ubuntu/Notebooks/rnn_experiments_2.json', 'w') as f:
        json.dump(experiments, f)

