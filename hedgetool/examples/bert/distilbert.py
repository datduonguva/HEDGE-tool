import os
import numpy as np
import tensorflow as tf
import json
from transformers import DistilBertTokenizer, TFDistilBertModel
from hedgetool import HEDGE

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
    
    # build the model and load the pretrain weights 
    tf.keras.backend.clear_session()
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=["acc"],
    )
    model.load_weights(MODEL_ROOT)

    # get the pretrained tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # this is the string of interest
    sentence = ['i think the song sound very interesting']
    encoded_sentence = tokenizer(sentence, return_tensors='np')['input_ids'][0:1, 1:-1]

    max_length = len(encoded_sentence[0])
    padding = 0


    # extend the HEDGE class and define how to make the inference based on the encoded input
    class MyHEDGE(HEDGE):
        def __init__(self, *arg):
            super(MyHEDGE, self).__init__(*arg)

        def predict_prob(self, encoded_sentence):

            # For Bert-type of model, encoded sentence must be prepended with a [CLS] token
            # and appended with a [PAD] token

            # index for [CLS] is 101, index for [PAD] is 102
            encoded_sentence = np.hstack(
                [
                    np.array([101]*len(encoded_sentence)).reshape((-1, 1)),
                    encoded_sentence,
                    np.array([102]*len(encoded_sentence)).reshape((-1, 1))
                ]
            )
            attention_mask = np.ones(encoded_sentence.shape)
            return self.model.predict(
                {'input_ids_pl': encoded_sentence, 'attention_mask_pl': attention_mask}
            )

    hedge = MyHEDGE(model, max_length, padding, tokenizer.ids_to_tokens)
    P_history, spans, contributions = hedge.main_algorithm(encoded_sentence)
    hedge.visualize(encoded_sentence, P_history, contributions)


if __name__ == '__main__':
    main()
