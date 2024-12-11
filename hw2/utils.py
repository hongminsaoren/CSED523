import string
import torch
import sacrebleu
import json
import sentencepiece
import numpy as np

#######################
### For IBM Model 1 ###
#######################

def preprocess(sentence):
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))  # strip punctuation
    return sentence.strip().lower().split()

def visualize_alignment(alignment, source_sentence, target_sentence):
    # Create an alignment matrix
    alignment_matrix = [[' ' for _ in target_sentence] for _ in source_sentence]
    for (i, j) in alignment:
        alignment_matrix[i][j] = '✔'

    # Print the matrix with source and target words
    print("Alignment Matrix:")
    print("      " + "  ".join(f"{t:>5}" for t in target_sentence))
    for i, row in enumerate(alignment_matrix):
        print(f"{source_sentence[i]:>5} " + "  ".join(f"{cell:>5}" for cell in row))

#######################
### For Seq2Seq NMT ###
#######################


# vocaburary and essential definitions
# you can make this part as comment for SMT implementation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab = sentencepiece.SentencePieceProcessor()
vocab.Load("multi30k.model")
pad_id = vocab.PieceToId("<pad>")
bos_id = vocab.PieceToId("<s>")
eos_id = vocab.PieceToId("</s>")



# TODO: Implement prediction (decoding) algorithm for your NMT model
# Hint: We have 2 options for decoding: greedy search and beam search
#      Please revisit the lecture slides to choose better decoding algorithm

def predict_greedy(model, sentences, max_length=100):
    tensors = []
    for sentence in sentences:
        indices = vocab.EncodeAsIds(sentence)
        indices_augmented = [bos_id] + indices + [eos_id]
        tensors.append(torch.tensor(indices_augmented, dtype=torch.long, device=device))
    batch = torch.nn.utils.rnn.pad_sequence(tensors, padding_value=pad_id)
    max_seq_len, batch_size = batch.shape
    encoder_output, encoder_mask, hidden = model.encode(batch)
    decoder_input = torch.full((1, batch_size), bos_id).to(device)
    sequence_ids = decoder_input.cpu().numpy().T
    for i in range(max_length):
        logits, hidden, attention_weights = model.decode(decoder_input, hidden, encoder_output, encoder_mask)
        decoder_input = logits.argmax(dim=-1)
        sequence_ids = np.hstack((sequence_ids, decoder_input.cpu().numpy().T))
    sequence = [vocab.decode_ids(sentence_ids.tolist()) for sentence_ids in sequence_ids]
    return sequence  

def predict_beam():
  pass

# adjust method according to your implementation
# default is greedy
def evaluate(model, dataset, batch_size=64, method="greedy"):
    assert method in {"greedy", "beam"}
    source_sentences = [example[0] for example in dataset]
    target_sentences = [' '.join(example[1]) for example in dataset]
    model.eval()
    predictions = []
    with torch.no_grad():
        for start_index in range(0, len(source_sentences), batch_size):
          if method == "greedy":
            prediction_batch = predict_greedy(model, source_sentences[start_index:start_index + batch_size])
          else:
            prediction_batch = predict_beam(model, source_sentences[start_index:start_index + batch_size])
            prediction_batch = [candidates[0] for candidates in prediction_batch]
          predictions.extend(prediction_batch)
    return sacrebleu.corpus_bleu(predictions, [target_sentences]).score

def get_raw_predictions(model, dataset, method="greedy", batch_size=64):
  assert method in {"greedy", "beam"}
  source_sentences = [example[0] for example in dataset]
  model.eval()
  predictions = []
  with torch.no_grad():
    for start_index in range(0, len(source_sentences), batch_size):
      if method == "greedy":
        prediction_batch = predict_greedy(
            model, source_sentences[start_index:start_index + batch_size])
      else:
        prediction_batch = predict_beam(
            model, source_sentences[start_index:start_index + batch_size])
      predictions.extend(prediction_batch)
  return predictions

def generate_predictions_file_for_submission(filepath, model, dataset, bleu_score):
  models = {"attention": model}
  datasets = {"test": dataset}
  methods = ["greedy"] # you can adjust according to your implementation
  predictions = {}
  for model_name, model in models.items():
    for dataset_name, dataset in datasets.items():
      for method in methods:
        print(
            "Getting predictions for {} model on {} set using {} "
            "search...".format(model_name, dataset_name, method))
        if model_name not in predictions:
          predictions[model_name] = {}
        if dataset_name not in predictions[model_name]:
          predictions[model_name][dataset_name] = {}
        try:
          predictions[model_name][dataset_name][method] = get_raw_predictions(
              model, dataset, method)
        except:
          print("!!! WARNING: An exception was raised, setting predictions to None !!!")
          predictions[model_name][dataset_name][method] = None
  print("Writing predictions to {}...".format(filepath))
  with open(filepath, "w") as outfile:
    json.dump(predictions, outfile, indent=2, ensure_ascii=False)
    # Write BLEU score to the end of the file
    outfile.write("\n")
    outfile.write("BLEU: {:.2f}".format(bleu_score))
    
  print("Finished writing predictions to {}.".format(filepath))