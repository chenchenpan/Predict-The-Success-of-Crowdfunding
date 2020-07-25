import unittest
from encoder import Encoder, Mapping, open_glove
import pandas as pd
import numpy as np
import json

def get_fake_dataset(with_text_col=False, text_only=False): 
## you can change this to create your own test dataset here ##
    if with_text_col:
        df_train = pd.DataFrame({'height': [1,2,3], 'key_words': ['hello', 'hi', 'yes'], 
                         'text': ["Strange Wit, an original graphic novel about Jane Bowles",
                                  "The true biography of the historical figure, writer, alcoholic, lesbian",
                                  "world traveler: Jane Sydney Auer Bowles."], 
                         'label': [0, 1, 2]})
        df_dev = pd.DataFrame({'height': [4,7,5], 'key_words': ['hi', 'hi', 'yes'],
                               'text': ["FAM is the new mobile app which combines events and all your social media needs", 
                                        "Destiny, NY - FINAL HOURS!",
                                        "A graphic novel about two magical ladies in love."], 
                               'label': [1, 1, 2]})
        df_test = pd.DataFrame({'height': [2,5,3], 'key_words': ['hello', 'yes', 'yes'],
                        'text':["Publishing Magus Magazine,We are publishing a magazine that focuses on the folklore of the occult and paranormal.",
                                "It is tabloid format but with academic articles",
                                "a strong-willed Russian madam and The Cross at its most fabulous."],
                        'label': [2, 1, 2]})
        if text_only:
            metadata = {'output_type': 'classes',
                        'input_features': ['text'],
                        'output_label': ['label'],
                        'input_text': ['text'],
                        'input_bool': [],
                        'input_categorical': [],
                        'input_datetime': [],
                        'input_int': [],
                        'input_float': []
                        } 

        else:
            metadata = {'output_type': 'classes',
                        'input_features': ['height','key_words','text'],
                        'output_label': ['label'],
                        'input_text': ['text'],
                        'input_bool': [],
                        'input_categorical': ['key_words'],
                        'input_datetime': [],
                        'input_int': ['height'],
                        'input_float': []
                        } 
    else:    
        df_train = pd.DataFrame({'height': [1,2,3], 'key_words': ['hello', 'hi', 'yes'], 'label': [0, 1, 2]})
        df_dev = pd.DataFrame({'height': [4,7,5], 'key_words': ['hi', 'hi', 'yes'], 'label': [1, 1, 2]})
        df_test = pd.DataFrame({'height': [2,5,3], 'key_words': ['hello', 'yes', 'yes'], 'label': [2, 1, 2]})
        metadata = {'output_type': 'classes',
                    'input_features': ['height','key_words'],
                    'output_label': ['label'],
                    'input_text': [],
                    'input_bool': [],
                    'input_categorical': ['key_words'],
                    'input_datetime': [],
                    'input_int': ['height'],
                    'input_float': []
                    }
    return df_train, df_dev, df_test, metadata



    
class TestEncoder(unittest.TestCase):
    def test_strucdata_only(self):
        df_train, df_dev, df_test, metadata = get_fake_dataset(with_text_col=False)
        encoder = Encoder(metadata, text_config=None)

        y_train, X_train, _ = encoder.fit_transform(df_train)
        y_dev, X_dev, _ = encoder.transform(df_dev)
        y_test, X_test, _ = encoder.transform(df_test)

        X_train_true = np.array([
            [-1.22474487,  1.        ,  0.        ,  0.        ], 
            [ 0.        ,  0.        ,  1.        ,  0.        ], 
            [ 1.22474487,  0.        ,  0.        ,  1.        ]])
        y_train_true = np.array([
            [1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
        # print(X_train)
        self.assertTrue(np.isclose(X_train_true, X_train).all())
        self.assertTrue(np.isclose(y_train_true, y_train).all())
        X_dev_true = np.array([
            [2.44948974, 0.        , 1.        , 0.        ],
            [6.12372436, 0.        , 1.        , 0.        ],
            [3.67423461, 0.        , 0.        , 1.        ]])
        y_dev_true = np.array([
            [0., 1., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
        self.assertTrue(np.isclose(X_dev_true, X_dev).all())
        self.assertTrue(np.isclose(y_dev_true, y_dev).all())
        X_test_true = np.array([
            [0.        , 1.        , 0.        , 0.        ],
            [3.67423461, 0.        , 0.        , 1.        ],
            [1.22474487, 0.        , 0.        , 1.        ]])
        y_test_true = np.array([
            [0., 0., 1.],
           [0., 1., 0.],
           [0., 0., 1.]])
        self.assertTrue(np.isclose(X_test_true, X_test).all())
        self.assertTrue(np.isclose(y_test_true, y_test).all())


    def test_tfidf(self):
        df_train, df_dev, df_test, metadata = get_fake_dataset(with_text_col=True)

        text_config = Mapping()
        text_config.mode = 'tfidf'
        text_config.max_words = 20
        print('*' * 20)
        print(text_config.mode)

        encoder = Encoder(metadata, text_config=text_config)
        y_train, X_train_struc, X_train_text = encoder.fit_transform(df_train)
        y_dev, X_dev_struc, X_dev_text = encoder.transform(df_dev)
        y_test, X_test_struc, X_test_text = encoder.transform(df_test)

        X_train_text_true = np.array([
            [0.        , 0.69314718, 0.69314718, 0.        , 0.91629073,
            0.91629073, 0.91629073, 0.91629073, 0.91629073, 0.91629073,
            0.91629073, 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 1.55141507, 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.91629073, 0.91629073, 0.91629073, 0.91629073,
            0.91629073, 0.91629073, 0.91629073, 0.91629073, 0.        ],
           [0.        , 0.69314718, 0.69314718, 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.91629073]])
        X_train_struc_true = np.array([
            [-1.22474487,  1.        ,  0.        ,  0.        ],
            [0.        ,  0.        ,  1.        ,  0.        ],
            [1.22474487,  0.        ,  0.        ,  1.        ]])
        self.assertTrue(np.isclose(X_train_text_true, X_train_text).all())
        self.assertTrue(np.isclose(X_train_struc_true, X_train_struc).all())
        X_dev_text_true = np.array([
            [0.        , 0.        , 0.        , 0.91629073, 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.91629073, 0.91629073,
            0.91629073, 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ]])
        X_dev_struc_true = np.array([
            [2.44948974, 0.        , 1.        , 0.        ],
            [6.12372436, 0.        , 1.        , 0.        ],
            [3.67423461, 0.        , 0.        , 1.        ]])
        self.assertTrue(np.isclose(X_dev_text_true, X_dev_text).all())
        self.assertTrue(np.isclose(X_dev_struc_true, X_dev_struc).all())

        X_test_text_true = np.array([
            [0.        , 0.        , 0.        , 1.55141507, 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.91629073, 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.91629073, 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ]])
        X_test_struc_true = np.array([
            [0.        , 1.        , 0.        , 0.        ],
            [3.67423461, 0.        , 0.        , 1.        ],
            [1.22474487, 0.        , 0.        , 1.        ]])
        self.assertTrue(np.isclose(X_test_text_true, X_test_text).all())
        self.assertTrue(np.isclose(X_test_struc_true, X_test_struc).all())


    def test_word_embedding(self):
        df_train, df_dev, df_test, metadata = get_fake_dataset(with_text_col=True)

        glove_file_path = 'glove.6B.50d.txt'# need be changed to where you store the pre-trained GloVe file.
        
        text_config = Mapping()
        text_config.mode = 'glove'
        text_config.max_words = 20
        text_config.maxlen = 5
        text_config.embedding_dim = 50
        text_config.embeddings_index = open_glove(glove_file_path)

        encoder = Encoder(metadata, text_config=text_config)
        y_train, X_train_struc, X_train_text = encoder.fit_transform(df_train)
        y_dev, X_dev_struc, X_dev_text = encoder.transform(df_dev)
        y_test, X_test_struc, X_test_text = encoder.transform(df_test)

        X_train_text_true = np.array([
            [ 9, 10, 11,  2,  3],
            [15, 16, 17, 18, 19],
            [1,  2, 1, 1,  3]])
        X_train_struc_true = np.array([
            [-1.22474487,  1.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  1.        ,  0.        ],
            [ 1.22474487,  0.        ,  0.        ,  1.        ]])
        self.assertTrue(np.isclose(X_train_text_true, X_train_text).all())
        self.assertTrue(np.isclose(X_train_struc_true, X_train_struc).all())
        X_dev_text_true = np.array([
            [1, 1, 1, 1, 1],
           [1, 1, 1, 1,  0],
           [1, 1, 1, 1, 1]])
        X_dev_struc_true = np.array([
            [2.44948974, 0.        , 1.        , 0.        ],
            [6.12372436, 0.        , 1.        , 0.        ],
            [3.67423461, 0.        , 0.        , 1.        ]])
        # print(X_dev_text)
        self.assertTrue(np.isclose(X_dev_text_true, X_dev_text).all())
        self.assertTrue(np.isclose(X_dev_struc_true, X_dev_struc).all())
        X_test_text_true = np.array([
            [14,  4, 1, 1, 1],
           [1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1]])
        X_test_struc_true = np.array([
            [0.        , 1.        , 0.        , 0.        ],
            [3.67423461, 0.        , 0.        , 1.        ],
            [1.22474487, 0.        , 0.        , 1.        ]])
        self.assertTrue(np.isclose(X_test_text_true, X_test_text).all())
        self.assertTrue(np.isclose(X_test_struc_true, X_test_struc).all())


if __name__ == '__main__':
    unittest.main()