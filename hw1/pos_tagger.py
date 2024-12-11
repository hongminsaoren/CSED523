from multiprocessing import Pool
import numpy as np
import time
import csv
from utils import *



""" Contains the part of speech tagger class. """


def evaluate(data, model):
    """Evaluates the POS model on some sentences and gold tags.

    This model can compute a few different accuracies:
        - whole-sentence accuracy
        - per-token accuracy
        - compare the probabilities computed by different styles of decoding

    You might want to refactor this into several different evaluation functions,
    or you can use it as is. 
    
    As per the write-up, you may find it faster to use multiprocessing (code included). 
    
    """
    processes = 4
    sentences = data[0]
    tags = data[1]
    n = len(sentences)
    k = n//processes
    n_tokens = sum([len(d) for d in sentences])
    unk_n_tokens = sum([1 for s in sentences for w in s if w not in model.word2idx.keys()])
    predictions = {i:None for i in range(n)}
    probabilities = {i:None for i in range(n)}
         
    start = time.time()
    pool = Pool(processes=processes)
    res = []
    for i in range(0, n, k):
        res.append(pool.apply_async(infer_sentences, [model, sentences[i:i+k], i]))
    ans = [r.get(timeout=None) for r in res]
    predictions = dict()
    for a in ans:
        predictions.update(a)
    print(f"Inference Runtime: {(time.time()-start)/60} minutes.")
    
    start = time.time()
    pool = Pool(processes=processes)
    res = []
    for i in range(0, n, k):
        res.append(pool.apply_async(compute_prob, [model, sentences[i:i+k], tags[i:i+k], i]))
    ans = [r.get(timeout=None) for r in res]
    probabilities = dict()
    for a in ans:
        probabilities.update(a)
    print(f"Probability Estimation Runtime: {(time.time()-start)/60} minutes.")


    token_acc = sum([1 for i in range(n) for j in range(len(sentences[i])) if tags[i][j] == predictions[i][j]]) / n_tokens
    unk_token_acc = sum([1 for i in range(n) for j in range(len(sentences[i])) if tags[i][j] == predictions[i][j] and sentences[i][j] not in model.word2idx.keys()]) / unk_n_tokens
    whole_sent_acc = 0
    num_whole_sent = 0
    for k in range(n):
        sent = sentences[k]
        eos_idxes = indices(sent, '.')
        start_idx = 1
        end_idx = eos_idxes[0]
        for i in range(1, len(eos_idxes)):
            whole_sent_acc += 1 if tags[k][start_idx:end_idx] == predictions[k][start_idx:end_idx] else 0
            num_whole_sent += 1
            start_idx = end_idx+1
            end_idx = eos_idxes[i]
    print("Whole sent acc: {}".format(whole_sent_acc/num_whole_sent))
    print("Mean Probabilities: {}".format(sum(probabilities.values())/n))
    print("Token acc: {}".format(token_acc))
    print("Unk token acc: {}".format(unk_token_acc))
    
    confusion_matrix(pos_tagger.tag2idx, pos_tagger.idx2tag, predictions.values(), tags, 'cm.png')

    return whole_sent_acc/num_whole_sent, token_acc, sum(probabilities.values())/n


class POSTagger():
    def __init__(self):
        """Initializes the tagger model parameters and anything else necessary. """
        ## TODO
        self.data = None
        self.all_tags = None
        self.all_words = None
        self.tag2idx = None
        self.idx2tag = None
        self.word2idx = None
        self.unigrams = None
        self.bigrams = None
        self.trigrams = None
        self.emissions = None
    
    
    def get_unigrams(self):
        """
        Computes unigrams. 
        Tip. Map each tag to an integer and store the unigrams in a numpy array. 
        """
        ## TODO
        tag_counts = np.zeros(len(self.all_tags)) # Count of each tag
        for tags in self.data[1]:  # Iterate over all tags
            for tag in tags:
                tag_idx = self.tag2idx[tag]
                tag_counts[tag_idx] += 1
        self.unigrams = tag_counts / np.sum(tag_counts)

    def get_bigrams(self):        
        """
        Computes bigrams. 
        Tip. Map each tag to an integer and store the bigrams in a numpy array
             such that bigrams[index[tag1], index[tag2]] = Prob(tag2|tag1). 
        """
        ## TODO
        tag_counts_2d = np.zeros((len(self.all_tags), len(self.all_tags)) )
        tag_counts = np.zeros(len(self.all_tags))
        for tags in self.data[1]:  # Iterate over all tags
            for tag in tags:
                tag_idx = self.tag2idx[tag]
                tag_counts[tag_idx] += 1
            for i in range(1, len(tags)):
                tag1_idx = self.tag2idx[tags[i-1]]
                tag2_idx = self.tag2idx[tags[i]]
                tag_counts_2d[tag1_idx, tag2_idx] += 1
        for i in range(len(self.all_tags)): # Normalize
            row_sum = np.sum(tag_counts_2d[i])
            if row_sum != 0:
                tag_counts_2d[i] /= row_sum
        self.bigrams = tag_counts_2d
            
    
    def get_trigrams(self):
        """
        Computes trigrams. 
        Tip. Similar logic to unigrams and bigrams. Store in numpy array. 
        """
        ## TODO
        tag_counts_3d = np.zeros((len(self.all_tags), len(self.all_tags), len(self.all_tags))) 
        for tags in self.data[1]:  # Iterate over all tags
            for i in range(2, len(tags)):
                tag1_idx = self.tag2idx[tags[i-2]]
                tag2_idx = self.tag2idx[tags[i-1]]
                tag3_idx = self.tag2idx[tags[i]]
                tag_counts_3d[tag1_idx, tag2_idx, tag3_idx] += 1
        for i in range(len(self.all_tags)): # Normalize
            for j in range(len(self.all_tags)):
                row_sum = np.sum(tag_counts_3d[i][j])
                if row_sum != 0:
                    tag_counts_3d[i, j] /= row_sum
        self.trigrams = tag_counts_3d
    
    def get_emissions(self):
        """
        Computes emission probabilities. 
        Tip. Map each tag to an integer and each word in the vocabulary to an integer. 
             Then create a numpy array such that lexical[index(tag), index(word)] = Prob(word|tag) 
        """
        ## TODO
        self.all_words = list(set([word for sentence in self.data[0] for word in sentence]))
        self.word2idx = {self.all_words[i]:i for i in range(len(self.all_words))}
        emissions = np.zeros((len(self.all_tags), len(self.all_words)))
        tag_counts = np.zeros(len(self.all_tags))
        for i, sentence in enumerate(self.data[0]):
            tags = self.data[1][i]
            for j, word in enumerate(sentence):
                tag_idx = self.tag2idx[tags[j]]
                word_idx = self.word2idx.get(word, None)
                tag_counts[tag_idx] += 1
                if word_idx is not None:
                    emissions[tag_idx, word_idx] += 1
        for i in range(len(self.all_tags)):
            if tag_counts[i] != 0:
                emissions[i] /= tag_counts[i]
        self.emissions = emissions

    def train(self, data):
        """Trains the model by computing transition and emission probabilities."""
        self.data = data
        self.all_tags = list(set([t for tag in data[1] for t in tag]))
        self.tag2idx = {self.all_tags[i]:i for i in range(len(self.all_tags))}
        self.idx2tag = {v:k for k,v in self.tag2idx.items()}
        
        # Compute unigrams, bigrams, trigrams, emissions
        self.get_unigrams()
        self.get_bigrams()
        self.get_trigrams()
        self.get_emissions()

    def sequence_probability(self, sequence, tags):
        """Computes the probability of a tagged sequence given the emission/transition
        probabilities.
        """
        log_prob = 0.0
        num_tags = len(self.all_tags)
        for i in range(len(sequence)):
            word = sequence[i]
            tag = tags[i]
            word_idx = self.word2idx.get(word, None)
            tag_idx = self.tag2idx[tag]

            # Emission probability
            if word_idx is not None:
                emission_prob = self.emissions[tag_idx, word_idx]
            else:
                emission_prob = 1e-6  # Small probability for unknown words
            if emission_prob == 0:
                emission_prob = 1e-6
            log_prob += np.log(emission_prob)

            # Transition probability
            if i == 0:
                trans_prob = self.unigrams[tag_idx]
            else:
                prev_tag_idx = self.tag2idx[tags[i - 1]]
                trans_prob = self.bigrams[prev_tag_idx, tag_idx]
            if trans_prob == 0:
                trans_prob = 1e-6
            log_prob += np.log(trans_prob)

        return log_prob

    def inference(self, sequence):
        """Tags a sequence with part of speech tags using greedy decoding."""
        tags = []
        num_tags = len(self.all_tags)

        for i in range(len(sequence)):
            word = sequence[i]
            word_idx = self.word2idx.get(word, None)
            
            # For unknown words
            if word_idx is not None:
                emission_probs = self.emissions[:, word_idx]
            else:
                emission_probs = np.ones(num_tags) / num_tags  # Uniform probability

            # For the first word, use unigram probabilities
            if i == 0:
                probs = self.unigrams * emission_probs
            else:
                prev_tag_idx = self.tag2idx[tags[i - 1]]
                trans_probs = self.bigrams[prev_tag_idx, :]
                probs = trans_probs * emission_probs

            # Choose the tag with the highest probability
            tag_idx = np.argmax(probs)
            tags.append(self.idx2tag[tag_idx])

        return tags


if __name__ == "__main__":
    pos_tagger = POSTagger()

    train_data = load_data("data/train_x.csv", "data/train_y.csv")
    dev_data = load_data("data/dev_x.csv", "data/dev_y.csv")
    test_data = load_data("data/test_x.csv")

    pos_tagger.train(train_data)

    # Experiment with your decoder using greedy decoding.
    evaluate(dev_data, pos_tagger)
    
    # Predict tags for the test set
    test_predictions = []
    for sentence in test_data:
        test_predictions.extend(pos_tagger.inference(sentence))
    
    processed_test_ids = [i for i in range(len(test_predictions))]
    
    # Write predictions to a CSV file "data/test_y.csv"
    with open("data/test_y.csv", "w", newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(['id', 'tag'])
        for idx, tag in zip(processed_test_ids, test_predictions):
            writer.writerow([idx, tag])
