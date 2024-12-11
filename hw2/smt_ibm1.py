import pickle
import tqdm
from itertools import product
from utils import preprocess
from utils import visualize_alignment
import math
import numpy as np
class IBMModel1:
    def __init__(self, data, num_iterations=10, epsilon=1.0, compute_perplexity=True):
        self.data = data # aligned corpus as shown above
        self.num_iterations = num_iterations # iterations of expectation-maximization
        self.epsilon = epsilon
        self.compute_perplexity = compute_perplexity
        
        # Preprocess bitext data:
        self.source_words, self.target_words = set(), set()
        for (source,target) in self.data:
            self.source_words.update(source)
            self.target_words.update(target)
        
        # Initialize uniform probabilities:
        self.translation_probs = {(s,t): 1.0/len(self.target_words)
                                  for s,t in product(self.source_words, self.target_words)}
        
    def e_step(self):
        # YOUR SOLUTION HERE
        # - Iterate over paired sentences in the data and compute:
        # - (1) counts, the number of times a source word is translated into a target word,
        #       weighted by alignment probabilities
        # - (2) total, the sum of counts over all possible target words
        # TODO
        counts = {(s,t): 0.0 for s,t in product(self.source_words, self.target_words)}
        total = {t: 0.0 for t in self.target_words}
        for source, target in self.data:
            for s, t in zip(source, target):
                counts[(s,t)] += self.translation_probs[(s,t)]
                total[t] += self.translation_probs[(s,t)]
        return counts, total
        
    def m_step(self, counts, total):
        # YOUR SOLUTION HERE
        # - Update self.translation_probs using counts and total
        # TODO
        for s, t in self.translation_probs:
            if total[t] == 0:
                self.translation_probs[(s,t)] = 0
            else:
                self.translation_probs[(s,t)] = counts[(s,t)] / total[t]
        return self.translation_probs
        
    def train(self):
        # Run EM for self.num_iterations:
        for idx in tqdm.tqdm(range(self.num_iterations)):
            if self.compute_perplexity: 
                print("Iteration: {} | Perplexity: {}".format(idx, self.perplexity()))
            counts, total = self.e_step()
            self.m_step(counts, total)
        if self.compute_perplexity:
            print("Iteration: {} | Perplexity: {}".format(self.num_iterations, self.perplexity()))

    def probability(self, source, target):
        # YOUR SOLUTION HERE
        # - Use the normalization trick from lecture to efficiently compute probabilities
        # - We'll use self.epsilon here, which is defined in the initialization
        # TODO
        count = 1
        for s, t in zip(source, target):
            count *= self.translation_probs[(s,t)]
        mult = 1
        for t in target:
            prob_t = 0
            for s in source:
                prob_t += self.translation_probs[(s,t)]
            mult *= prob_t
        if mult == 0:
            return self.epsilon
        return count/mult
        
    def perplexity(self):
        # YOUR SOLUTION HERE
        # - Iterate over each pair of sentences in the dataset
        # - Call self.probability and compute a sum in log space
        # - Feel free to comment this out while testing your initial model
        # TODO
        log_prob = []
        for source, target in self.data:
            log_prob.append(math.log(self.probability(source, target), 2))

        return 2 ** -np.mean(log_prob)
        
    def get_alignment(self, source, target):
        # YOUR SOLUTION HERE
        # - Find the best word alignment for a source, target pair
        # - Output a list of [(source_idx, target_idx)]
        #   For example: (["a", "book"], ["ein", "buch"])
        #   should have an alignment [(0,0), (1,1)]
        # TODO
        alignments = []
        for idx_s, s in enumerate(source):
            alignment = ()
            max_prob = -math.inf
            for idx_t, t in enumerate(target):
                if (s,t) not in self.translation_probs:
                    continue
                prob = self.translation_probs[(s,t)]
                if prob > max_prob:
                    alignment = (idx_s, idx_t)
                    max_prob = prob
            if max_prob > 0:
                alignments.append(alignment)

        return alignments

if __name__ == "__main__":
    # Define the paths to your local data files
    train_src_path = 'multi30k/train.en'
    train_tgt_path = 'multi30k/train.de'

    # Read the data from the files
    with open(train_src_path, 'r', encoding='utf-8') as f_src, open(train_tgt_path, 'r', encoding='utf-8') as f_tgt:
        src_sentences = f_src.readlines()
        tgt_sentences = f_tgt.readlines()

    # Preprocess the data and create aligned sentence pairs
    aligned_data = []
    for src_line, tgt_line in zip(src_sentences[:1000], tgt_sentences[:1000]):
        source = preprocess(src_line)
        target = preprocess(tgt_line)
        aligned_data.append((source, target))

    # Train the IBM Model 1
    ibm = IBMModel1(aligned_data, compute_perplexity=True)
    ibm.train()

    # Visualize the alignment for a sample sentence pair
    ## TODO see the visualize_alignment function in utils.py
            
    for i in range(10):
        source_sentence = aligned_data[i][0]
        target_sentence = aligned_data[i][1]
        alignment = ibm.get_alignment(source_sentence, target_sentence)
        visualize_alignment(alignment, source_sentence, target_sentence)
    ## End of implementation
    # Save the translation probabilities
    student_id = "49005121"
    with open(f"results/smt_{student_id}.pkl", "wb") as outfile:
        pickle.dump(ibm.translation_probs, outfile, protocol=pickle.HIGHEST_PROTOCOL)