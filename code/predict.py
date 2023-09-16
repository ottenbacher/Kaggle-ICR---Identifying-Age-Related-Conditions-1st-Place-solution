import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers as R
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras import layers as L
from tensorflow.keras import optimizers as O
from tensorflow.keras import constraints as C
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
tf.keras.utils.set_random_seed(722)


with open('../SETTINGS.json') as json_file:
    settings = json.load(json_file)

TEST_DATA_CLEAN_PATH = settings['TEST_DATA_CLEAN_PATH']
SAVED_MODEL_WEIGHTS_BEST_DIR = settings['SAVED_MODEL_WEIGHTS_BEST_DIR']
SUBMISSION_DIR = settings['SUBMISSION_DIR']

test_df = pd.read_csv(TEST_DATA_CLEAN_PATH, index_col='Id')

X_test = test_df.values

@tf.keras.utils.register_keras_serializable()
def smish(x):
    return x * K.tanh(K.log(1 + K.sigmoid(x)))


@tf.keras.utils.register_keras_serializable()
class GatedLinearUnit(L.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.linear = L.Dense(units)
        self.sigmoid = L.Dense(units, activation="sigmoid")
        self.units = units

    def get_config(self):
        config = super().get_config()
        config['units'] = self.units
        return config
    
    def call(self, inputs):
        return self.linear(inputs) * self.sigmoid(inputs)
    

@tf.keras.utils.register_keras_serializable()
class GatedResidualNetwork(L.Layer):
    def __init__(self, units, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.relu_dense = L.Dense(units, activation=smish)
        self.linear_dense = L.Dense(units)
        self.dropout = L.Dropout(dropout_rate)
        self.gated_linear_unit = GatedLinearUnit(units)
        self.layer_norm = L.LayerNormalization()
        self.project = L.Dense(units)

    def get_config(self):
        config = super().get_config()
        config['units'] = self.units
        config['dropout_rate'] = self.dropout_rate
        return config
    
    def call(self, inputs):
        x = self.relu_dense(inputs)
        x = self.linear_dense(x)
        x = self.dropout(x)
        if inputs.shape[-1] != self.units:
            inputs = self.project(inputs)
        x = inputs + self.gated_linear_unit(x)
        x = self.layer_norm(x)
        return x
    

@tf.keras.utils.register_keras_serializable()
class VariableSelection(L.Layer):
    def __init__(self, num_features, units, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.grns = list()
        # Create a GRN for each feature independently
        for idx in range(num_features):
            grn = GatedResidualNetwork(units, dropout_rate)
            self.grns.append(grn)
        # Create a GRN for the concatenation of all the features
        self.grn_concat = GatedResidualNetwork(units, dropout_rate)
        self.softmax = L.Dense(units=num_features, activation="softmax")
        self.num_features = num_features
        self.units = units
        self.dropout_rate = dropout_rate

    def get_config(self):
        config = super().get_config()
        config['num_features'] = self.num_features
        config['units'] = self.units
        config['dropout_rate'] = self.dropout_rate
        return config
    
    def call(self, inputs):
        v = L.concatenate(inputs)
        v = self.grn_concat(v)
        v = tf.expand_dims(self.softmax(v), axis=-1)

        x = []
        for idx, input_ in enumerate(inputs):
            x.append(self.grns[idx](input_))
        x = tf.stack(x, axis=1)

        outputs = tf.squeeze(tf.matmul(v, x, transpose_a=True), axis=1)
        return outputs
    

@tf.keras.utils.register_keras_serializable()
class VariableSelectionFlow(L.Layer):
    def __init__(self, num_features, units, dropout_rate, dense_units=None, **kwargs):
        super().__init__(**kwargs)
        self.variableselection = VariableSelection(num_features, units, dropout_rate)
        self.split = L.Lambda(lambda t: tf.split(t, num_features, axis=-1))
        self.dense = dense_units
        if dense_units:
            self.dense_list = [L.Dense(dense_units, \
                                       activation='linear') \
                               for _ in tf.range(num_features)
                              ]
        self.num_features = num_features
        self.units = units
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        
    def get_config(self):
        config = super().get_config()
        config['num_features'] = self.num_features
        config['units'] = self.units
        config['dropout_rate'] = self.dropout_rate
        config['dense_units'] = self.dense_units
        return config        
    
    def call(self, inputs):
        split_input = self.split(inputs)
        if self.dense:
            l = [self.dense_list[i](split_input[i]) for i in range(len(self.dense_list))]
        else:
            l = split_input
        return self.variableselection(l)        
    
    
MODELS_WEIGHTS = os.listdir(SAVED_MODEL_WEIGHTS_BEST_DIR)

y_pred = np.zeros_like(test_df.iloc[:,0].values, dtype=np.float32)
batch_size = 32
units_1 = 32
drop_1 = 0.75
dense_units = 8
units_2 = 16
drop_2 = 0.5
units_3 = 8
drop_3 = 0.25
    
inputs_1 = tf.keras.Input(shape=(56,))
        
features_1 = VariableSelectionFlow(56, units_1, drop_1, dense_units=dense_units)(inputs_1)
features_2 = VariableSelectionFlow(units_1, units_2, drop_2)(features_1)         
features_3 = VariableSelectionFlow(units_2, units_3, drop_3)(features_2)         

outputs = L.Dense(1, activation="sigmoid")(features_3)

model = Model(inputs=inputs_1, outputs=outputs)
for n, model_weights in enumerate(MODELS_WEIGHTS):
      
    model.load_weights(SAVED_MODEL_WEIGHTS_BEST_DIR + model_weights)
    y_pred += model.predict(X_test)[:,0]
    
y_pred /= len(MODELS_WEIGHTS)
p1 = y_pred
p0 = 1 - p1
p = np.stack([p0, p1]).T
class_0_est_instances = p[:,0].sum()
others_est_instances = p[:,1:].sum()
new_p = p * np.array([[1/(class_0_est_instances if i==0 else others_est_instances) for i in range(p.shape[1])]])
new_p = new_p / np.sum(new_p, axis=1, keepdims=1)

test_df[['class_0', 'class_1']] = new_p
test_df[['class_0', 'class_1']].to_csv(SUBMISSION_DIR + 'submission.csv')
