import torch

def train_torch_net(trainloader,neural_net,loss_function, n_epochs = 50):

    optimizer = torch.optim.Adam(neural_net.parameters(), lr=1e-4)

    # Run the training loop
    for epoch in range(0, n_epochs):  # 5 epochs at maximum

        # Set current loss value
        loss_per_batch = []

        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):

            # Get and prepare inputs
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            #targets = targets.reshape((targets.shape[0], 1))

            # Zero the gradients
            optimizer.zero_grad()
            # Perform forward pass
            outputs = neural_net(inputs)
            # Compute loss
            loss = loss_function(outputs, targets)
            # Perform backward pass
            loss.backward()
            # Perform optimization
            optimizer.step()

            loss_per_batch.append(loss.item())

    return neural_net, loss_per_batch