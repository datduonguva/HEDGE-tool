import os
import json
import copy
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt

class HEDGE:
    def __init__(self, model, max_length, padding=-1, ids_to_token=None):
        """
        Args:
            model: Tensorflow's keras model
            max_length: The length of the sentence that is required by the model
            padding:    The index of the token that is used for padding the sentence to the
            maximum length. It will also be the token used to mask the tokens when
            calculating the text interaction scores and well as contribution scores
            ids_to_token: (optional) a dictionary mapping encoded sentence back to text.
        """
        self.model = model
        self.padding = padding
        self.max_length = max_length
        self.ids_to_token = ids_to_token

    def gamma(self, encoded_sentence, label, S, j1, j2):
        """
        model   : the keras model to make prediction
        label   : the categorical index of the prediction
        encoded_sentence       : the encoded sentence of shape (1, max_length)
        S, j1, j2 are lists  
        
        this implements equation 3
        """
        all_feature_sets = set(range(self.max_length))

        j1 = set(j1)
        j2 = set(j2)
        S = set(S)

        x_prime = np.tile(encoded_sentence, (4, 1))
        x_prime[0, list(all_feature_sets - (S| j1 | j2))] = self.padding
        x_prime[1, list(all_feature_sets - (S | j1))] = self.padding
        x_prime[2, list(all_feature_sets - (S | j2))] = self.padding
        x_prime[3, list(all_feature_sets - S)] = self.padding
        
        # only care about the interested category (given by `label`)
        expectations = self.predict_prob(x_prime)[:, label]
        return expectations[0] - expectations[1] - expectations[2] + expectations[3]
       
    def phi(self, encoded_sentence, label, P, j1, j2):
        """
        P: a list of list
        j1, j2: each of them are list whose set(j1) | set(j2) belong to P
        encoded_sentence: encoded_sentence of shape (1, sentence length)
        """

        N = [element for element in P if j1[0] != element[0]]

        j1_and_j2 = j1 + j2
        
        score = 0
        for S_size in range(len(N)+1):
            weight = (
                np.math.factorial(S_size) *
                np.math.factorial(len(P) - S_size - 1) /
                np.math.factorial(len(P))
            )
            score = 0 
            for S in list(combinations(N, S_size)):
                # S is a list of text spans
                S_flatten = [token for spand in S for token in spand]
                
                score += weight*self.gamma(encoded_sentence, label, S_flatten, j1, j2)

        return score

    def predict_prob(self, *args, **kwargs):
        raise NotImplementedError("prediction function has not been implemented")

    def main_algorithm(self, encoded_sentence):
        """
        encoded_sentence: a 2-D numpy array of shapep (1, None)
        """
        
        # get the model prediction for the given encoded_sentence
        label = np.argmax(self.predict_prob(encoded_sentence)[0])
        
        padding_start_index = -1

        for i in range(self.max_length -1, 0, -1):
            if encoded_sentence[0][i] != self.padding:
                padding_start_index = i
                break


        P_history = []
        
        # the first step contains the whole sentence
        P_history.append([list(range(padding_start_index + 1))])

        for step in range(1, padding_start_index + 1):
            
            P = P_history[-1]
            # loop through all text span in P:
            # temporarily store value in phi_array to find the minum later
            min_span_index = -1 
            min_j_index = -1
            min_phi_value = 1e8 
            for spand_index, span in enumerate(P):
                if len(span) == 1:
                    continue
                for j in range(1, len(span)):
                    j1 = span[:j]
                    j2 = span[j:]

                    phi_value = self.phi(encoded_sentence, label, P, j1, j2)

                    if phi_value < min_phi_value:
                        min_phi_value = phi_value
                        min_span_index = spand_index
                        min_j_index = j
            
            new_P = copy.deepcopy(P)
            span = new_P.pop(min_span_index)
            j1, j2 = span[:min_j_index], span[min_j_index:]

            new_P.insert(min_span_index, j1)
            new_P.insert(min_span_index + 1, j2)
            P_history.append(new_P)

        spans = [span for P in P_history for span in P]
        contribution = self.get_contribution_score(encoded_sentence, spans, label)

        return P_history, spans, contribution

    def get_contribution_score(self, x, spans, label):
        """
        this function receives a list of spands and returns the contribution scores to the label
        """
        masked_input = np.ones((len(spans), len(x[0])))*self.padding

        for i, span in enumerate(spans):
            masked_input[i, span]  = x[0][span]

        outputs = self.predict_prob(masked_input)
        predicted_label_probs = outputs[:, label].copy()

        # make the output on that label really small
        # so that the max becomes the of the other classes
        outputs[:, label] = -1e10
        other_probs = np.max(outputs, axis=1)
        
        return predicted_label_probs - other_probs

    def visualize(self, encoded_sentence, P_history, contributions):
        """
        visualize the model's explanation
        """
        spans = [span for P in P_history for span in P]
        
        span_contrib_mapping = {str(span): contrib for span, contrib in zip(spans, contributions)}

        scores = np.zeros((P_history[0][0][-1]+1, P_history[0][0][-1]+1))

        for i, P in enumerate(P_history):
            for span in P:
                span_str = str(span)
                for token in span:
                    scores[i, token] = span_contrib_mapping[span_str]
        ax = plt.gca()
        
        if self.ids_to_token != None:
            tokens = [self.ids_to_token[i] for i in encoded_sentence[0]]
        else:
            tokens = [str(i) for i in encoded_sentence[0]]
        im = ax.imshow(scores, cmap="coolwarm")
        for i in range(len(P_history)):
            for j in range(max(P_history[0][0]) + 1):
                ax.text(j-0.3, i, tokens[j])
        plt.show()

