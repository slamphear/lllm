import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Data.tokenizer import tokenize_text


# Create batches
def create_batches(token_indices, batch_size, seq_length):
    # Calculate the total number of full sequences we can make
    num_seq = len(token_indices) // seq_length

    # Trim the list of token indices to fit evenly into full sequences
    token_indices = token_indices[: num_seq * seq_length]

    # Convert the list of token indices into a tensor and reshape it into sequences
    tensor_data = torch.tensor(token_indices).view(-1, seq_length)

    # The input data is all sequences except the last one for each sequence
    input_data = tensor_data[:-1]

    # The target data is all sequences except the first one, shifted by one token forward
    target_data = tensor_data[1:]

    # Create batches from input and target data
    num_batches = len(input_data) // batch_size
    input_batches = input_data[: num_batches * batch_size].view(
        num_batches, batch_size, seq_length
    )
    target_batches = target_data[: num_batches * batch_size].view(
        num_batches, batch_size, seq_length
    )

    return input_batches, target_batches


def train_model(
    model,
    vocab,
    num_epochs,
    learning_rate,
    scheduler_patience,
    scheduler_factor,
    input_batches,
    target_batches,
):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, "min", patience=scheduler_patience, factor=scheduler_factor
    )

    # Training loop
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}:\nBatch ", end="")
        epoch_loss = 0  # To keep track of loss in an epoch

        for i in range(input_batches.size(0)):
            print(f"{i+1}/{input_batches.size(0)}...", end="", flush=True)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(input_batches[i])

            # Compute loss
            loss = criterion(output.view(-1, len(vocab)), target_batches[i].view(-1))

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Compute average epoch loss
        average_epoch_loss = epoch_loss / input_batches.size(0)

        # Step the scheduler
        scheduler.step(average_epoch_loss)

        # Optionally: print loss and learning rate at each epoch
        print(
            f'Epoch {epoch+1}/{num_epochs}, Loss: {average_epoch_loss}, Learning rate: {optimizer.param_groups[0]["lr"]}'
        )
