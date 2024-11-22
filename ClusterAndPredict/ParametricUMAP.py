import pickle
import time
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, MultiHeadAttention, LayerNormalization, Add, Layer
from umap.parametric_umap import ParametricUMAP
import numpy as np
import random

# Patch for TensorFlow
from tensorflow.python.keras.engine import data_adapter

def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)

data_adapter._is_distributed_dataset = _is_distributed_dataset


# -----------------------------------------------------------------------------
# MLP
# -----------------------------------------------------------------------------

class ParametricUMAPEncoder:
    def __init__(self, num_components, embedding_np, y_tensor, trained=False, seed=23):
        self.num_components = num_components
        self.embedding_np = embedding_np
        self.y_tensor = y_tensor
        self.trained = trained
        self.seed = seed
        
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        print("seed:", self.seed)

        # Convert to tensor
        self.embedding_np = tf.convert_to_tensor(self.embedding_np)
        self.y_tensor = tf.convert_to_tensor(self.y_tensor)

        # Define the enhanced encoder network
        self.encoder = keras.Sequential([
            layers.InputLayer(input_shape=(3072,)),
            layers.Dense(units=512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(units=256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(units=128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(units=64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
            layers.BatchNormalization(),
            layers.Dense(units=self.num_components)
        ])

        # Initialize reducer with the encoder
        self.reducer = ParametricUMAP(encoder=self.encoder, n_components=self.num_components)

        # Load weights if already trained
        if self.trained:
            self._load_weights()
        else:
            self.fit()

    def fit(self):
        start_time = time.time()
        self.reducer.fit(self.embedding_np, y=self.y_tensor)
        self.encoder.save_weights('encoder.weights.h5')
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Training time: {execution_time} seconds")

    def transform(self):
        start_time = time.time()
        embedding_np = self.reducer.transform(self.embedding_np)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Transforming time: {execution_time} seconds")
        return embedding_np

    def _load_weights(self):
        self.encoder.load_weights('encoder.weights.h5')

# # -----------------------------------------------------------------------------
# # Attention
# # -----------------------------------------------------------------------------

# # 自定义扩展维度的 Keras 层
# class ExpandDimsLayer(Layer):
#     def call(self, inputs):
#         return tf.expand_dims(inputs, axis=1)

# # 自定义去除维度的 Keras 层
# class SqueezeLayer(Layer):
#     def call(self, inputs):
#         return tf.squeeze(inputs, axis=1)


# class ParametricUMAPEncoder:
#     def __init__(self, num_components, embedding_np, y_tensor, trained=False):
#         self.num_components = num_components
#         self.embedding_np = embedding_np
#         self.y_tensor = y_tensor
#         self.trained = trained

#         # Convert to tensor
#         self.embedding_np = tf.convert_to_tensor(self.embedding_np)
#         self.y_tensor = tf.convert_to_tensor(self.y_tensor)

# # -----------------------------------------------------------------------------


#         # Define the enhanced encoder network with attention
#         input_layer = Input(shape=(3072,))
        
#         # 添加扩展维度层以适应 Transformer 层
#         x = ExpandDimsLayer()(input_layer)
        
#         x = Dense(units=512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)
#         x = BatchNormalization()(x)
#         x = Dropout(0.2)(x)
#         x = Dense(units=256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)
#         x = BatchNormalization()(x)
#         x = Dropout(0.2)(x)

#         # Attention block
#         attention_output = MultiHeadAttention(num_heads=4, key_dim=x.shape[-1])(x, x)
#         x = Add()([x, attention_output])  # Residual connection
#         x = LayerNormalization()(x)

#         x = Dense(units=128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)
#         x = BatchNormalization()(x)
#         x = Dropout(0.2)(x)
#         x = Dense(units=64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)
#         x = BatchNormalization()(x)

#         # Another Attention block
#         attention_output = MultiHeadAttention(num_heads=4, key_dim=x.shape[-1])(x, x)
#         x = Add()([x, attention_output])  # Residual connection
#         x = LayerNormalization()(x)

#         # 添加去除维度层
#         x = SqueezeLayer()(x)

# # -----------------------------------------------------------------------------


#         # Final dense layer to match the number of components
#         output_layer = Dense(units=self.num_components, activation=None)(x)

#         self.encoder = Model(inputs=input_layer, outputs=output_layer)
        
#         self.encoder.summary()

#         # Initialize reducer with the encoder
#         self.reducer = ParametricUMAP(encoder=self.encoder, n_components=self.num_components)

#         # Load weights if already trained
#         if self.trained:
#             self._load_weights()
#         else:
#             self.fit()


#     def fit(self):
#         start_time = time.time()
#         self.reducer.fit(self.embedding_np, y=self.y_tensor)
#         self.encoder.save_weights('encoder.weights.h5')
#         end_time = time.time()
#         execution_time = end_time - start_time
#         print(f"Training time: {execution_time} seconds")

#     def transform(self):
#         start_time = time.time()
#         embedding_np = self.reducer.transform(self.embedding_np)
#         end_time = time.time()
#         execution_time = end_time - start_time
#         print(f"Transforming time: {execution_time} seconds")
#         return embedding_np

#     def _load_weights(self):
#         self.encoder.load_weights('encoder.weights.h5')

# # Example usage
# if __name__ == "__main__":
#     # Load embedding_np and y_tensor
#     with open('variables.pkl', 'rb') as f:
#         embedding_np, y_tensor = pickle.load(f)

#     num_components = 100
#     trained = False  # Set to True if the model has already been trained

#     param_umap_encoder = ParametricUMAPEncoder(num_components, embedding_np, y_tensor, trained=trained)
#     embedding_np_transformed = param_umap_encoder.transform()
