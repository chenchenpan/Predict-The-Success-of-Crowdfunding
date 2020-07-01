import unittest
from encoder import Encoder
import pandas as pd
import numpy as np

    
class TestEncoder(unittest.TestCase):
    def test_strucdata_only(self):
        df_train = pd.DataFrame({'height': [1,2,3], 'key_words': ['hello', 'hi', 'yes'], 'label': [1, 2, 3]})
        df_dev = pd.DataFrame({'height': [4,7,5], 'key_words': ['hi', 'hi', 'yes'], 'label': [2, 2, 3]})
        df_test = pd.DataFrame({'height': [2,5,3], 'key_words': ['hello', 'yes', 'yes'], 'label': [3, 2, 3]})
        metadata = {'input_features': ['height','key_words'],
                    'output_label': ['label'],
                    'input_text': [],
                    'input_bool': [],
                    'input_categorical': ['key_words'],
                    'input_datetime': [],
                    'input_int': ['height'],
                    'input_float': []
                    } 

        # text_config=TFIDFEncodeConfig(100)

        encoder = Encoder(metadata, text_config=None)

        y_train, X_train, _ = encoder.fit_transform(df_train)
        y_dev, X_dev, _ = encoder.transform(df_dev)
        y_test, X_test, _ = encoder.transform(df_test)

        X_train_true = np.array([
            [-1.22474487,  1.        ,  0.        ,  0.        ], 
            [ 0.        ,  0.        ,  1.        ,  0.        ], 
            [ 1.22474487,  0.        ,  0.        ,  1.        ]])

        y_train_true = np.array([[1],[2],[3]])

        print(X_train)

        self.assertTrue(np.isclose(X_train_true., X_train).all())
        self.assertTrue(np.isclose(y_train_true, y_train).all())


        X_dev_true = np.array([
            [2.44948974, 0.        , 1.        , 0.        ],
            [6.12372436, 0.        , 1.        , 0.        ],
            [3.67423461, 0.        , 0.        , 1.        ]])

        y_dev_true = np.array([[2], [2], [3]])

        self.assertTrue(np.isclose(X_dev_true, X_dev).all())
        self.assertTrue(np.isclose(y_dev_true, y_dev).all())

        X_test_true = np.array([
            [0.        , 1.        , 0.        , 0.        ],
            [3.67423461, 0.        , 0.        , 1.        ],
            [1.22474487, 0.        , 0.        , 1.        ]])

        y_test_true = np.array([[3], [2], [3]])

        self.assertTrue(np.isclose(X_test_true, X_test).all())
        self.assertTrue(np.isclose(y_test_true, y_test).all())



    def test_tfidf(self):
        pass


    def test_word_embedding(self):
        pass

if __name__ == '__main__':
    unittest.main()