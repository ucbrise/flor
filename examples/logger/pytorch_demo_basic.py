#Source: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/linear_regression/main.py
import flor 
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

log = flor.log 

# Toy dataset
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)



# Hyper-parameters
@flor.track_execution
def main():
    input_size = 1
    output_size = 1
    num_epochs = 60
    learning_rate = 0.001

    # Linear regression model
    model = nn.Linear(log.parameter(input_size), log.parameter(output_size))

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=log.parameter(learning_rate))  

    # Train the model
    for epoch in range(log.parameter(num_epochs)):
        # Convert numpy arrays to torch tensors
        inputs = torch.from_numpy(x_train)
        targets = torch.from_numpy(y_train)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 5 == 0:
            #evaluate_loss(epoch, loss.item())
            log.metric(epoch)
            log.metric(loss.item())
            #print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    # Plot the graph
    #predicted = model(torch.from_numpy(x_train)).detach().numpy()
    #plt.plot(x_train, y_train, 'ro', label='Original data')
    #plt.plot(x_train, predicted, label='Fitted line')
    #plt.legend()
    #plt.show()

    # Save the model checkpoint
    #torch.save(model.state_dict(), 'model.ckpt')
with flor.Context('pytorch_demo'):
    main()










