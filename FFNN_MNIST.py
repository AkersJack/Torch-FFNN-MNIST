import torch # Root package
import torch.nn as nn # Neural Networks
import torchvision # Vision Datasets, Architectures & Transforms 
import torchvision.transforms as transforms # Image Transformations 
import torch.optim as optim # Optimizers (gradient descent, ADAM, etc.)
import matplotlib.pyplot as plt # Plotting/Graphing 

    
# Shows examples of the images in the MNIST dataset
def showFigure(example_data, example_targets):
    fig = plt.figure()
    plt.tight_layout()

    # 6 example images
    for i in range(6):
        plt.subplot(2, 3, i + 1) # 2 rows of 3 images
        plt.imshow(example_data[i][0], cmap = "gray", interpolation = "none")
        plt.title(f"Ground Truth: {example_targets[i]}")
        plt.xticks([])
        plt.yticks([])
    plt.show() # show the figure


def showFigure_pred(example_data, example_targets):
    with torch.no_grad():
        test_data = example_data.reshape(-1, 28 * 28) 
        output = model(test_data.to(device)) # Load images into the model 
    
    fig = plt.figure()
    fig.tight_layout()

    # Show 28 images
    for i in range(28):
        plt.subplot(4, 7, i + 1) # 4 rows of 7 images
        plt.imshow(example_data[i][0], cmap = 'gray', interpolation = 'none')
        plt.title(f"Prediction: {output.data.max(1, keepdim = True)[1][i].item()}")
        plt.xlabel(f"Ground Truth: {example_targets[i]}")
        plt.xticks([])
        plt.yticks([])
    plt.show() # show the figure

# The Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) # First layer [784, 100]
        self.relu = nn.ReLU() # Rectified Linear Unit (ReLU)
        self.l2 = nn.Linear(hidden_size, num_classes) # Second Layer (output layer) [100, 10]
    
    # Forward pass
    def forward(self, x): # x is [100, 10]
        out = self.l1(x)  #  In [100, 10] out [100, 100]
        out = self.relu(out) # In [100, 100] out [100, 100]
        out = self.l2(out) # In [100, 100] out [100, 10]          
        return out # Output shape [100, 10]

# Training the model 
def train(epoch):
    model.train() # Prepare the model for training
    for batch, (images, labels) in enumerate(train_loader): 
        # Origin Shape: [100, 1, 28, 28]
        # Resized Shape: [100, 784]
        images = images.reshape(-1, 28 * 28).to(device) 
        labels = labels.to(device) 

        # Forward Pass
        output = model(images) # Run the Forward pass
        loss = lossFunc(output, labels) # Calculate the loss

        # Backpropagation
        optimizer.zero_grad() # Reset the gradients of all optimized torch.Tensor
        loss.backward()
        optimizer.step() # Take a step (gradient descent)

        # Log/output training progress
        if batch % log_interval == 0:
            print('Epoch [{}/{}], Step [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, num_epochs, batch * len(images), len(train_loader.dataset),
                100. * batch / len(train_loader), loss.item()))

            train_losses.append(loss.item())
            train_counter.append((batch * 100) + ((epoch - 1) * len(train_loader.dataset)))

            # Save model and optimizer to continue training if necessary
            torch.save(model.state_dict(), './results/model.pth') 
            torch.save(optimizer.state_dict(), './results/optimizer.pth')

# Test the model
def test():
    model.eval()
    n_correct = 0
    test_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            # Forward Pass
            output = model(images)
            test_loss += lossFunc_test(output, labels).item()
            
            _, pred = torch.max(output.data, 1) # Returns (value, index)
            n_correct += (pred == labels).sum().item()
    
    test_loss /= len(test_loader.dataset) # Average the total loss
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, n_correct, len(test_loader.dataset),
        100. * n_correct / len(test_loader.dataset)))

# Show the performance of the network as it trains over time. 
def showPerformance():
    fig = plt.figure
    plt.plot(train_counter, train_losses, color = "blue")
    plt.scatter(test_counter, test_losses, color = "red")
    plt.legend(['Train Loss', 'Test Loss'], loc = 'upper right')
    plt.xlabel('Number of training examples seen')
    plt.ylabel('Cross-entropy loss (log loss)')
    plt.show()



if __name__ == "__main__":
    # Hyperparameters
    input_size = 784 # 28 x 28 
    hidden_size = 100 # hidden layer size
    num_classes = 10 # 0 - 9
    num_epochs = 2 # Each epoch is a round of training 
    batch_size = 100 # Number of training examples utilized in one iteration
    learning_rate = 0.001 # step size 
    log_interval = 10 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     
    # Training Data
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./files/', train = True, download = True,
                                   transform = torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                       ])),
        batch_size = batch_size, shuffle = True
    )
    
    # Test Data
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./files/', train = False, download = True,
                                   transform = torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                       ])),
        batch_size = batch_size, shuffle = False
    )

    # Data Representation
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(num_epochs + 1)]


    # Load a set of example images and labels
    examples = enumerate(test_loader)
    batch, (samples, labels) = next(examples)
    
    

    # To show examples of the images
    showFigure(samples, labels)

    # Create the Neural Network object
    model = NeuralNetwork(input_size, hidden_size, num_classes)

    # If using cuda it will load the model and its associated parameters into the computing device's memory
    model = model.to(device)
    
    # Loss and Optimizer
    lossFunc = nn.CrossEntropyLoss()
    lossFunc_test = nn.CrossEntropyLoss(reduction = "sum")
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    # Training Loop
    test() # Baseline test with no training 
    for epoch in range(1, num_epochs + 1):
        train(epoch)
        test() 

    # Show the models performance over the training period
    showPerformance()

    # Show Images and the networks predictions
    showFigure_pred(samples, labels)









