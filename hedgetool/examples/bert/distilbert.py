import os
import numpy as np
import tensorflow as tf
import json
from transformers import DistilBertTokenizer, TFDistilBertModel

MODEL_ROOT = "/home/datduong/gdrive/projects/c12_distilBERT/02"

def build_model():
    pretrained_model = TFDistilBertModel.from_pretrained(
        'distilbert-base-uncased', output_attentions=False
    )
    input_ids = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, name='input_ids_pl'
    )
    attention_mask = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, name='attention_mask_pl'
    )
    
    # get the output of the '[CLS'] token on the last layer
    bert_output = pretrained_model(
        {'input_ids': input_ids, 'attention_mask': attention_mask},
        return_dict=True
    )['last_hidden_state'][:, 0]

    pre_classification = tf.keras.layers.Dense(128, activation='tanh')(bert_output)
    dropout_1 = tf.keras.layers.Dropout(0.3)(pre_classification)

    classification_output = tf.keras.layers.Dense(2, activation='softmax')(dropout_1)

    model = tf.keras.models.Model(
        inputs=[input_ids, attention_mask], outputs=classification_output
    )
    return model


def main():
    tf.keras.backend.clear_session()
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=["acc"],
    )
    model.load_weights(MODEL_ROOT)
    model.summary()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    sentence = 'sentence = 'i think the song sound very interesting'
