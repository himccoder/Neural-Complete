import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import re

# ===================== Dataset =====================
class CharDataset(Dataset):
    def __init__(self, data, sequence_length, stride, vocab_size):
        self.data = data
        self.sequence_length = sequence_length
        self.stride = stride
        self.vocab_size = vocab_size
        self.sequences = []
        self.targets = []
        
        # Create overlapping sequences with stride
        for i in range(0, len(data) - sequence_length, stride):
            self.sequences.append(data[i:i + sequence_length])
            self.targets.append(data[i + 1:i + sequence_length + 1])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        target = torch.tensor(self.targets[idx], dtype=torch.long)

        return sequence, target

# ===================== Model =====================
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_dim=30):
        super().__init__()   #Initialize a recurrent neural network (RNN) model for character-level text generation.
        # input_size: Number of unique characters in the vocabulary
        # hidden_size: Number of features in the hidden state of the RNN    
        self.hidden_size = hidden_size
        ## Embedding layer to map input indices to dense vectors 
        self.embedding = nn.Embedding(output_size, embedding_dim)
        # TODO: Initialize your model parameters as needed e.g. W_e, W_h, etc.
        # Weight matrix for transforming input embeddings to hidden state space
        self.W_xh = nn.Parameter(torch.randn(embedding_dim, hidden_size) * 0.01)
        # Weight matrix for transforming the previous hidden state to the current hidden state
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        # Bias vector for the hidden state computation
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        # Weight matrix for projecting the hidden state to the output space (vocabulary size)
        self.W_hy = nn.Parameter(torch.randn(hidden_size, output_size) * 0.01)
        # Bias vector for the output logits
        self.b_y = nn.Parameter(torch.zeros(output_size))



    def forward(self, x, hidden):
        """
        x in [b, l] # b is batch_size and l is sequence length
        """
        x_embed = self.embedding(x)  # [b=batch_size, l=sequence_length, e=embedding_dim]
        b, l, _ = x_embed.size()
        x_embed = x_embed.transpose(0, 1) # [l, b, e]
        if hidden is None or hidden.size(0) != b:
            # If hidden is None or batch size has changed, initialize hidden state
            h_t_minus_1 = self.init_hidden(b)
        else:
            h_t_minus_1 = hidden
        output = []
        for t in range(l):
            #  TODO: Implement forward pass for a single RNN timestamp, append the hidden to the output 
            x_t = x_embed[t]                       # shape: [batch_size, embedding_dim]
            h_t = torch.tanh(x_t @ self.W_xh + h_t_minus_1 @ self.W_hh + self.b_h)  # [b, hidden_size]
            output.append(h_t)
            h_t_minus_1 = h_t
        
        output = torch.stack(output) # [l, b, e]
        output = output.transpose(0, 1) # [b, l, e]
        
        # TODO set these values after completing the loop above
        # Note: You can use the output of the last hidden state to compute the final output
        
        logits = output @ self.W_hy + self.b_y    # [b, l, vocab_size]
        final_hidden = h_t.clone()

        return logits, final_hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(device)

# ===================== Training =====================
device = 'cpu'
print(f"Using device: {device}")

def read_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read().lower()
        text = re.sub(r'[^a-z.,!?;:()\[\] ]+', '', text)
    return text

# To debug your model you should start with a simple sequence an RNN should predict this perfectly
#sequence = "abcdefghijklmnopqrstuvwxyz" * 100
sequence = read_file("warandpeace.txt") # Uncomment to read from file
vocab = sorted(set(sequence))
# TODO: Create a mapping from characters to indices
char_to_idx = {ch: idx for idx, ch in enumerate(vocab)}
# TODO: Create the reverse mapping 
idx_to_char = {idx: ch for ch, idx in char_to_idx.items()}

data = [char_to_idx[char] for char in sequence]

# TODO Understand and adjust the hyperparameters once the code is running
sequence_length = 50 # Length of each input sequence
stride = 3          # Stride for creating sequences
#embedding_dim = 2      # Dimension of character embeddings
#hidden_size = 1        # Number of features in the hidden state of the RNN
#learning_rate = 200    # Learning rate for the optimizer
learning_rate = 1e-3
hidden_size = 256
embedding_dim = 128

num_epochs = 1         # Number of epochs to train
batch_size = 200   # Batch size for training
vocab_size = len(vocab)
input_size = len(vocab)
output_size = len(vocab)

model = CharRNN(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

data_tensor = torch.tensor(data, dtype=torch.long)
train_size = int(len(data_tensor) * 0.9)
#TODO: Split the data into 90:10 ratio with PyTorch indexing
# Note: You can use the `torch.utils.data.random_split` function for this

train_data = data_tensor[:train_size]
test_data = data_tensor[train_size:]


train_dataset = CharDataset(train_data, sequence_length, stride, output_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    hidden = None
    for batch_inputs, batch_targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        #hidden = None  # <--- reset hidden state for each batch
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
        
        output, hidden = model(batch_inputs, hidden)
        # Detach removes the hidden state from the computational graph.
        # This is prevents backpropagating through the full history, and
        # is important for stability in training RNNs
        hidden = hidden.detach()

        

        # TODO compute the loss, backpropagate gradients, and update total_loss
        loss = criterion(output.view(-1, vocab_size), batch_targets.view(-1))  # Compute loss
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Backpropagate
        optimizer.step()  # Update model weights

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Test the model
# TODO: Implement a test loop to evaluate the model on the test set

# Create test dataset and loader
test_dataset = CharDataset(test_data, sequence_length, stride, output_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
print(f"Test set size: {len(test_dataset)}")
print(f"Test loader size (batches): {len(test_loader)}")

model.eval()
test_loss = 0.0
with torch.no_grad():
    hidden = None
    for batch_inputs, batch_targets in test_loader:
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
        output, hidden = model(batch_inputs, hidden)
        hidden = hidden.detach()
        loss = criterion(output.view(-1, vocab_size), batch_targets.view(-1))
        test_loss += loss.item()


if len(test_loader) > 0:
    print(f"Test Loss: {test_loss / len(test_loader):.4f}")
else:
    print("No test data available. Skipping test loss computation.")




# ===================== Text Generation =====================
def sample_from_output(logits, temperature=1.0):
    """
    Sample from the logits with temperature scaling.
    logits: Tensor of shape [batch_size, vocab_size] (raw scores, before softmax)
    temperature: a float controlling the randomness (higher = more random)
    """
    if temperature <= 0:
        temperature = 0.00000001
    # Apply temperature scaling to logits (increase randomness with higher values)
    scaled_logits = logits / temperature 
    # Apply softmax to convert logits to probabilities
    probabilities = F.softmax(scaled_logits, dim=1)
    
    # Sample from the probability distribution
    sampled_idx = torch.multinomial(probabilities, 1)
    return sampled_idx

def generate_text(model, start_text, n, k, temperature=1.0):
    """
        model: The trained RNN model used for character prediction.
        start_text: The initial string of length `n` provided by the user to start the generation.
        n: The length of the initial input sequence.
        k: The number of additional characters to generate.
        temperature: Optional
        A scaling factor for randomness in predictions. Higher values (e.g., >1) make 
            predictions more random, while lower values (e.g., <1) make predictions more deterministic.
            Default is 1.0.
    """
    #start_text = start_text.lower()
    #TODO: Implement the rest of the generate_text function
    # Hint: you will call sample_from_output() to sample a character from the logits
    model.eval()  # Set the model to evaluation mode
    generated_text = start_text.lower()

    # Convert start_text to a list of indices
    input_indices = [char_to_idx[c] for c in generated_text if c in char_to_idx]

    # Ensure input is exactly length n (pad or trim)
    if len(input_indices) < n:
        input_indices = [char_to_idx[' ']] * (n - len(input_indices)) + input_indices
    else:
        input_indices = input_indices[-n:]

    input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0)  # shape: [1, n]

    hidden = None  # Let model init hidden state

    with torch.no_grad():
        for _ in range(k):
            output, hidden = model(input_tensor, hidden)
            logits = output[:, -1, :]  # Get the logits for the last character
            next_char_idx = sample_from_output(logits, temperature=temperature)
            next_char = idx_to_char[next_char_idx.item()]
            generated_text += next_char

            # Prepare input for next step
            input_indices = input_indices[1:] + [next_char_idx.item()]
            input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0)

    return generated_text


# ===================== Main Loop =====================

print("Training complete. Now you can generate text.")
while True:
    start_text = input("Enter the initial text (n characters, or 'exit' to quit): ")
    
    if start_text.lower() == 'exit':
        print("Exiting...")
        break
    
    n = len(start_text) 
    k = int(input("Enter the number of characters to generate: "))
    temperature_input = input("Enter the temperature value (1.0 is default, >1 is more random): ")
    temperature = float(temperature_input) if temperature_input else 1.0
    
    completed_text = generate_text(model, start_text, n, k, temperature)
    
    print(f"Generated text: {completed_text}")