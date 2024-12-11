import matplotlib.pyplot as plt
import sentencepiece
import torch
import torch.nn as nn
from utils import preprocess, generate_predictions_file_for_submission
from trainutils import train
import numpy as np

class Seq2SeqNMTwithAttention(nn.Module):
  def __init__(self):
    super().__init__()

    # Initialize your model's parameters here. To get started, we suggest
    # setting all embedding and hidden dimensions to 256, using encoder and
    # decoder LSTMs with 2 layers, and using a dropout rate of 0.5.
    # Vocabulary size is  8000.

    # Implementation tip: To create a bidirectional LSTM, you don't need to
    # create two LSTM networks. Instead use nn.LSTM(..., bidirectional=True).
    
    
    # YOUR CODE HERE
    self.embedding_size = 256
    self.hidden_size = 256
    self.layers = 2
    self.dropout = 0.5
    self.embedding = nn.Embedding(len(vocab), self.embedding_size)
    self.encoder = nn.LSTM(self.embedding_size, self.hidden_size, self.layers, dropout=self.dropout, bidirectional=True)
    self.decoder = nn.LSTM(self.hidden_size, self.embedding_size, self.layers, dropout=self.dropout)
    self.output = nn.Linear(self.embedding_size, len(vocab))
    self.loss = nn.CrossEntropyLoss()

    # BEGIN SOLUTION
    
    # END SOLUTION

  def encode(self, source):
    """Encode the source batch using a bidirectional LSTM encoder.

    Args:
      source: An integer tensor with shape (max_source_sequence_length,
        batch_size) containing subword indices for the source sentences.

    Returns:
      A tuple with three elements:
        encoder_output: The output of the bidirectional LSTM with shape
          (max_source_sequence_length, batch_size, 2 * hidden_size).
        encoder_mask: A boolean tensor with shape (max_source_sequence_length,
          batch_size) indicating which encoder outputs correspond to padding
          tokens. Its elements should be True at positions corresponding to
          padding tokens and False elsewhere.
        encoder_hidden: The final hidden states of the bidirectional LSTM (after
          a suitable projection) that will be used to initialize the decoder.
          This should be a pair of tensors (h_n, c_n), each with shape
          (num_layers, batch_size, hidden_size). Note that the hidden state
          returned by the LSTM cannot be used directly. Its initial dimension is
          twice the required size because it contains state from two directions.
    """

    # Compute a tensor containing the length of each source sequence.
    lengths = torch.sum(source != pad_id, axis=0)

    embedding = self.embedding(source)
    out, (h_n, c_n) = self.encoder(embedding)
    h_n = h_n[:2] + h_n[2:]
    c_n = c_n[:2] + c_n[2:]
    encoder_mask = source == pad_id
    return out, encoder_mask, (h_n, c_n)


    # BEGIN SOLUTION
    
    # END SOLUTION

  def decode(self, decoder_input, initial_hidden, encoder_output, encoder_mask):
    """Run the decoder LSTM starting from an initial hidden state.

    The third and fourth arguments are not used in the baseline model, but are
    included for compatibility with the attention model in the next section.

    Args:
      decoder_input: An integer tensor with shape (max_decoder_sequence_length,
        batch_size) containing the subword indices for the decoder input. During
        evaluation, where decoding proceeds one step at a time, the initial
        dimension should be 1.
      initial_hidden: A pair of tensors (h_0, c_0) representing the initial
        state of the decoder, each with shape (num_layers, batch_size,
        hidden_size).
      encoder_output: The output of the encoder with shape
        (max_source_sequence_length, batch_size, 2 * hidden_size).
      encoder_mask: The output mask from the encoder with shape
        (max_source_sequence_length, batch_size). Encoder outputs at positions
        with a True value correspond to padding tokens and should be ignored.

    Returns:
      A tuple with three elements:
        logits: A tensor with shape (max_decoder_sequence_length, batch_size,
          vocab_size) containing unnormalized scores for the next-word
          predictions at each position.
        decoder_hidden: A pair of tensors (h_n, c_n) with the same shape as
          initial_hidden representing the updated decoder state after processing
          the decoder input.
        attention_weights: This will be implemented later in the attention
          model, but in order to maintain compatible type signatures, we also
          include it here. This can be None or any other placeholder value.
    """

    # These arguments are not used in the baseline model.

    # YOUR CODE HERE
    s_len, b_size = encoder_mask.shape
    d_len, _ = decoder_input.shape
    h_size = self.hidden_size
    embedding = self.embedding(decoder_input)
    encoder_mask = -1e9*encoder_mask.view(b_size, 1, s_len)
    d_out, (h_n, c_n) = self.decoder(embedding, initial_hidden)
    encoder_output = torch.sum(encoder_output.view(s_len, b_size, 2, h_size), dim=2)
    attn_logits = torch.einsum('dbh,sbh->dsb', [d_out, encoder_output])
    attn_weights = nn.functional.softmax(attn_logits, dim=1)
    context = torch.einsum('dsb,sbh->dbh', [attn_weights, encoder_output])
    logits = self.output(d_out + context)
    return logits.view(d_len, b_size, -1), (h_n, c_n), attn_weights.view(d_len, b_size, s_len)
    # BEGIN SOLUTION
    
    # END SOLUTION

  def compute_loss(self, source, target):
    """Run the model on the source and compute the loss on the target.

    Args:
      source: An integer tensor with shape (max_source_sequence_length,
        batch_size) containing subword indices for the source sentences.
      target: An integer tensor with shape (max_target_sequence_length,
        batch_size) containing subword indices for the target sentences.

    Returns:
      A scalar float tensor representing cross-entropy loss on the current batch.
    """

    # Implementation tip: don't feed the target tensor directly to the decoder.
    # To see why, note that for a target sequence like <s> A B C </s>, you would
    # want to run the decoder on the prefix <s> A B C and have it predict the
    # suffix A B C </s>.

    # YOUR CODE HERE
    encoder_output, encoder_mask, encoder_hidden = self.encode(source)
    decoder_output, decoder_hidden, attention_weights = self.decode(target[:-1], encoder_hidden, encoder_output, encoder_mask)
    target = target[1:]
    loss = self.loss(decoder_output.permute(0, 2, 1), target)
    return loss
    # BEGIN SOLUTION
    
    # END SOLUTION

if __name__ == "__main__":
  # use GPU if available
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  # Define the paths to your local data files
  train_src_path = 'multi30k/train.en'
  train_tgt_path = 'multi30k/train.de'
  validation_src_path = 'multi30k/val.en'
  validation_tgt_path = 'multi30k/val.de'
  test_src_path = 'multi30k/test.en'
  test_tgt_path = 'multi30k/test.de'

  # Read the data from the files
  with open(train_src_path, 'r', encoding='utf-8') as f_src, open(train_tgt_path, 'r', encoding='utf-8') as f_tgt:
      src_sentences = f_src.readlines()
      tgt_sentences = f_tgt.readlines()

  with open(validation_src_path, 'r', encoding='utf-8') as f_src, open(validation_tgt_path, 'r', encoding='utf-8') as f_tgt:
      val_src_sentences = f_src.readlines()
      val_tgt_sentences = f_tgt.readlines()
  
  with open(test_src_path, 'r', encoding='utf-8') as f_src, open(test_tgt_path, 'r', encoding='utf-8') as f_tgt:
      test_src_sentences = f_src.readlines()
      test_tgt_sentences = f_tgt.readlines()

  # Preprocess the data and create aligned sentence pairs
  training_data = []
  for src_line, tgt_line in zip(src_sentences, tgt_sentences):
      source = preprocess(src_line)
      target = preprocess(tgt_line)
      training_data.append((source, target))
  
  validation_data = []
  for src_line, tgt_line in zip(val_src_sentences, val_tgt_sentences):
      source = preprocess(src_line)
      target = preprocess(tgt_line)
      validation_data.append((source, target))
  
  test_data = []
  for src_line, tgt_line in zip(test_src_sentences, test_tgt_sentences):
      source = preprocess(src_line)
      target = preprocess(tgt_line)
      test_data.append((source, target))
  
  # Prepare for vocabulary
  vocab = sentencepiece.SentencePieceProcessor()
  vocab.Load("multi30k.model")
  print("Vocabulary size:", vocab.GetPieceSize())
  pad_id = vocab.PieceToId("<pad>")
  bos_id = vocab.PieceToId("<s>")
  eos_id = vocab.PieceToId("</s>")

  # You are welcome to adjust these parameters based on your model implementation.
  num_epochs = 20
  batch_size = 64

  # Create the model
  model = Seq2SeqNMTwithAttention().to(device)
  train(model, num_epochs, batch_size, "nmt_model.pt", training_data, validation_data)

  # Generate test set predictions
  student_id = "49005121"
  generate_predictions_file_for_submission(f"results/{student_id}_predictions.json", model, test_data, 0.0)

  # visualize the attention weights
  # You may find the following annotated heatmap tutorial helpful:
  # https://matplotlib.org/3.1.3/gallery/images_contours_and_fields/image_annotated_heatmap.html.
  
  # You may visualize decoder attention on gold source-target pairs from the validation data.
  # You do not need to run any inference. You can simply visualize the attention weights
  # A figure with attention map plots for 2 sentence pairs from the validation set 
  # (the method imshow, or equivalent, will likely be useful here).

  # YOUR CODE HERE
  def visualize_attention(model,source, target,vocab,save_path):
      # Please save your plot to the results/student_id_attention.png
      src_batch = []
      for s in source:
          indices = vocab.EncodeAsIds(s)
          src_batch.append(torch.tensor([bos_id] + indices + [eos_id]))
      source_batch=nn.utils.rnn.pad_sequence(src_batch, padding_value=pad_id).to(device)
      tgt_batch = []
      for t in target:
          indices = vocab.EncodeAsIds(t)
          tgt_batch.append(torch.tensor([bos_id] + indices + [eos_id]))
      target_batch=nn.utils.rnn.pad_sequence(src_batch, padding_value=pad_id).to(device)
      encoder_output, encoder_mask, hidden = model.encode(source_batch)
          
      # Decode the target sentence and get attention weights
      _, _, attn_weights = model.decode(target_batch, hidden, encoder_output, encoder_mask)
          
      # Squeeze the attention weights to remove unnecessary dimensions
      attn_weights = attn_weights.squeeze(1).detach().cpu().numpy()
      source_tokens = vocab.EncodeAsPieces(source)
      target_tokens = vocab.EncodeAsPieces(target)    
      fig, ax = plt.subplots(figsize=(8, 8))
      im = ax.imshow(attn_weights[:len(target_tokens), :len(source_tokens)])

          # Set ticks and labels
      ax.set_xticks(np.arange(len(source_tokens)))
      ax.set_yticks(np.arange(len(target_tokens)))
      ax.set_xticklabels(source_tokens)
      ax.set_yticklabels(target_tokens)

          # Rotate the tick labels
      plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

          # Set the title of the plot
      ax.set_title("Attention Heatmap")
          
          # Adjust layout and save the figure
      fig.tight_layout()
          
          # Save the plot to the specified path
      plt.savefig(save_path)
      plt.close()
      
      
      
   # Visualize attention for two sentence pairs from validation data
  # You can adjust index of validation_data to visualize different sentence pairs
  source_sentence1, target_sentence1 = validation_data[9]
  source_sentence2, target_sentence2 = validation_data[10]
  source_sentence3, target_sentence3 = validation_data[8]
  print(source_sentence1)
  print(' '.join(source_sentence1))
  visualize_attention(model, ' '.join(source_sentence1), ' '.join(target_sentence1), vocab, f"results/{student_id}_attention_1.png")
  visualize_attention(model, ' '.join(source_sentence2), ' '.join(target_sentence2), vocab, f"results/{student_id}_attention_2.png")
  visualize_attention(model, ' '.join(source_sentence3), ' '.join(target_sentence3), vocab, f"results/{student_id}_attention_3.png")

  # END SOLUTION