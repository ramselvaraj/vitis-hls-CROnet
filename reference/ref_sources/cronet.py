### Import Packages and  Module
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_msssim import ssim  # or ms_ssim if you want multi-scale
from torch.profiler import record_function
import pickle

# Set global font settings for Times New Roman with size 18
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 18
### Phase 1: Generate the data to train the CRONet
def top(nelx, nely, volfrac, penal, rmin, num_iterations = 20):
    # Storing key data to train the deeponet
    All_data = []

    # INITIALIZE
    x = np.ones((nely, nelx)) * volfrac
    loop = 0
    change = 1.0
    # START ITERATION
    if num_iterations is not None:
        while change > 0.01 and loop<num_iterations:
            x = base_code(x, nelx, nely, All_data, loop)
            loop += 1

    else:
        while change > 0.01 and loop<num_iterations:
            x = base_code(x, nelx, nely, All_data, loop)
            loop += 1

    return All_data

def base_code(x, nelx, nely, All_data, loop):
    # Initialize Fe_iter_data as a dictionary
    Fe_iter_data = {}
    xold = x.copy()
    # FE-ANALYSIS
    U, K, F, KE = FE(nelx, nely, x, penal)



    # Store relevant data in the dictionary
    Fe_iter_data['nelx'] = nelx
    Fe_iter_data['nely'] = nely
    Fe_iter_data['penal'] = penal
    Fe_iter_data['x'] = x.copy()
    Fe_iter_data['U'] = U.copy()
    Fe_iter_data['K'] = K.copy()
    Fe_iter_data['F'] = F.copy()
    Fe_iter_data['KE'] = KE.copy()

    All_data.append(Fe_iter_data)  # Append the dictionary to All_data

    # OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
    KE = lk()
    c = 0.0
    dc = np.zeros((nely, nelx))
    for ely in range(nely):
        for elx in range(nelx):
            n1 = (nely + 1) * (elx) + ely
            n2 = (nely + 1) * (elx + 1) + ely
            Ue = U[[2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n2 + 2, 2 * n2 + 3, 2 * n1 + 2, 2 * n1 + 3], 0]
            c += x[ely, elx] ** penal * Ue @ KE @ Ue
            dc[ely, elx] = -penal * x[ely, elx] ** (penal - 1) * Ue @ KE @ Ue

    # FILTERING OF SENSITIVITIES
    dc = check(nelx, nely, rmin, x, dc)

    # DESIGN UPDATE BY THE OPTIMALITY CRITERIA METHOD
    x = OC(nelx, nely, x, volfrac, dc)

    # PRINT RESULTS
    change = np.max(np.abs(x - xold))
    print(f' It.: {loop} Obj.: {c:.4f} Vol.: {np.mean(x):.3f} ch.: {change:.3f}')

    # PLOT DENSITIES
    plt.figure(figsize=(2, 1), dpi=100)  # Reduced figure size and increased dpi for faster plotting
    plt.subplot(1,2,2)
    plt.imshow(-x, cmap='gray')# interpolation='none')
    plt.axis('off')  # Disable axis for faster plotting
    plt.show()
    return x

# OPTIMALITY CRITERIA UPDATE
def OC(nelx, nely, x, volfrac, dc):
    l1, l2, move = 0, 100000, 0.2
    while (l2 - l1) > 1e-4:
        lmid = 0.5 * (l2 + l1)
        xnew = np.maximum(0.001, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / lmid)))))
        if np.sum(xnew) - volfrac * nelx * nely > 0:
            l1 = lmid
        else:
            l2 = lmid
    return xnew

# MESH-INDEPENDENCY FILTER
def check(nelx, nely, rmin, x, dc):
    dcn = np.zeros_like(dc)
    for i in range(nelx):
         for j in range(nely):
            sum_val = 0.0
            for k in range(max(i - round(rmin), 0), min(i + round(rmin), nelx)):
                for l in range(max(j - round(rmin), 0), min(j + round(rmin), nely)):
                    fac = rmin - np.sqrt((i - k) ** 2 + (j - l) ** 2)
                    sum_val += max(0.0, fac)
                    dcn[j, i] += max(0.0, fac) * x[l, k] * dc[l, k]
            dcn[j, i] /= x[j, i] * sum_val
    return dcn

# FE-ANALYSIS
def FE(nelx, nely, x, penal):
    KE = lk()
    K = lil_matrix((2 * (nelx + 1) * (nely + 1), 2 * (nelx + 1) * (nely + 1)))
    F = np.zeros((2 * (nelx + 1) * (nely + 1), 1))
    U = np.zeros((2 * (nelx + 1) * (nely + 1), 1))
    for elx in range(nelx):
        for ely in range(nely):
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edof = np.array([2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n2 + 2, 2 * n2 + 3, 2 * n1 + 2, 2 * n1 + 3])
            K[np.ix_(edof, edof)] += x[ely, elx] ** penal * KE
    # DEFINE LOADS AND SUPPORTS (HALF MBB-BEAM)
    F[1, 0] = -1
    fixeddofs = np.union1d(np.arange(0, 2 * (nely + 1), 2), np.array([2 * (nelx + 1) * (nely + 1) - 1]))
    alldofs = np.arange(2 * (nelx + 1) * (nely + 1))
    freedofs = np.setdiff1d(alldofs, fixeddofs)
    # SOLVE
    U[freedofs, 0] = spsolve(K[freedofs, :][:, freedofs], F[freedofs, 0])
    U[fixeddofs, 0] = 0
    return U, K, F, KE

# ELEMENT STIFFNESS MATRIX
def lk():
    E = 1.0
    nu = 0.3
    k = np.array([
        1/2 - nu / 6, 1/8 + nu / 8, -1/4 - nu / 12, -1/8 + 3 * nu / 8,
        -1/4 + nu / 12, -1/8 - nu / 8, nu / 6, 1/8 - 3 * nu / 8
    ])
    KE = E / (1 - nu ** 2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
    ])
    return KE

#parameters
nelx = 60
nely = 20
volfrac = 0.5
penal = 3.0
rmin= 1.5
history_length=10
All_data = top(nelx, nely, volfrac, penal, rmin, num_iterations = 20)

with open('FEA_train_data_'+str(nelx)+'x'+str(nely)+'.pkl', 'wb') as f:
    pickle.dump(All_data, f)
### Phase 2: Train the CRONet
gpu_times_branch_net = []
class DeepONet(nn.Module):
    def __init__(self, neurons=40, output_neurons=(nely + 1) * (nelx + 1) * 2):
        super(DeepONet, self).__init__()
        self.neurons = neurons
        self.output_neurons = output_neurons
        self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))

        # Defining the branch and trunk networks
        self.branch_output = self._branch_net()  # CRNN for branch network
        self.trunk_output1 = self._trunk_net1()
        self.trunk_output2 = self._trunk_net2()
        self.trunk_output3 = self._trunk_net3()
        self.trunk_output4 = self._trunk_net4()
        self.trunk_output5 = self._trunk_net5()

    def _branch_net(self):
        class CRNNBranch(nn.Module):
            def __init__(self, neurons, output_neurons):
                super(CRNNBranch, self).__init__()
                self.neurons = neurons
                self.output_neurons = output_neurons

                # with torch.autograd.profiler.record_function("CONV2d"):
                #   param1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)

                # Convolutional layers
                self.conv_layers1 = nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
                    nn.SiLU()
                )

                self.conv_layers2 = nn.Sequential(
                    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
                    nn.SiLU()
                )

                self.conv_layers3 = nn.Sequential(
                    nn.MaxPool2d(2)
                )

                self.conv_layers4 = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),  # Ensures spatial dims are reduced to 1x1
                )

                # RNN layer
                self.rnn = nn.RNN(
                    input_size=32,   # Output channels of CNN
                    hidden_size=64,  # Hidden size of the RNN
                    num_layers=1,
                    batch_first=True,
                    nonlinearity="tanh",
                )

                # Fully connected layers
                self.fc1 = nn.Sequential(
                    nn.Linear(64, self.neurons),
                    nn.SiLU()
                )

                self.fc2 = nn.Sequential(
                    nn.Linear(self.neurons, self.output_neurons)
                )

            def forward(self, x):
                """
                x: (batch_size, history, height, width)
                """
                batch_size, history, height, width = x.shape

                # (batch_size * history, 1, height, width)
                x = x.view(batch_size * history, 1, height, width)

                # CNN -> shape: [batch_size * history, 32, 1, 1]
                with torch.autograd.profiler.record_function("CONVLAYER1_BRNCHNET"):
                  cnn_out1 = self.conv_layers1(x)
                with torch.autograd.profiler.record_function("CONVLAYER2_BRNCHNET"):
                  cnn_out2 = self.conv_layers2(cnn_out1)
                with torch.autograd.profiler.record_function("CONVLAYER3_BRNCHNET"):
                  cnn_out3 = self.conv_layers3(cnn_out2)
                with torch.autograd.profiler.record_function("CONVLAYER4_BRNCHNET"):
                  cnn_out = self.conv_layers4(cnn_out3)

                # Reshape for RNN -> [batch_size, history, 32]
                cnn_out = cnn_out.view(batch_size, history, -1)

                # RNN -> [batch_size, history, 64]
                with torch.autograd.profiler.record_function("RNN_BRNCHNET"):
                  rnn_out, _ = self.rnn(cnn_out)
                # Take the last time-step
                last_rnn_out = rnn_out[:, -1, :]  # [batch_size, 64]

                # Fully connected -> [batch_size, output_neurons]
                with torch.autograd.profiler.record_function("fc1_BRNCHNET"):
                  fc1 = self.fc1(last_rnn_out)
                with torch.autograd.profiler.record_function("fc2_BRNCHNET"):
                  fc2 = self.fc2(fc1)
                return fc2

        return CRNNBranch(self.neurons, self.output_neurons)

    def _trunk_net1(self):
        trunk_output1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(3, 3, 2), stride=1, padding=1),
            nn.SiLU()
        )
        return trunk_output1

    def _trunk_net2(self):
        trunk_output2 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 2), stride=1, padding=1),
            nn.SiLU()
        )
        return trunk_output2

    def _trunk_net3(self):
        trunk_output3 = nn.Sequential(
            nn.AdaptiveAvgPool3d((15, 10, 1)),
            nn.Flatten()
        )
        return trunk_output3

    def _trunk_net4(self):
        trunk_output4 = nn.Sequential(
            nn.Linear(32 * 15 * 10 * 1, self.neurons),
            nn.SiLU()
        )
        return trunk_output4

    def _trunk_net5(self):
        trunk_output5 = nn.Sequential(nn.Linear(self.neurons, self.output_neurons)
        )
        return trunk_output5

    def forward(self, x1, t1):
        """
        x1: (batch_size, 10, 20, 60)  # 10 consecutive branch inputs
        x2: (batch_size, 21, 61, 2)   # trunk input
        """
        # Pass the branch data as is to the CRNN branch
        # No unsqueeze(1) for x1, because we already have shape (batch, history, height, width)
        x1 = self.branch_output(x1)

        # For the trunk, we need a channel dimension: (batch, 1, 21, 61, 2)
        t1 = t1.unsqueeze(1)
        with torch.autograd.profiler.record_function("CONVLAYER1_TRUNKNET"):
          t2 = self.trunk_output1(t1)
        with torch.autograd.profiler.record_function("CONVLAYER2_TRUNKNET"):
          t3 = self.trunk_output2(t2)
        with torch.autograd.profiler.record_function("CONVLAYER3_TRUNKNET"):
          t4 = self.trunk_output3(t3)
        with torch.autograd.profiler.record_function("CONVLAYER4_TRUNKNET"):
          t5 = self.trunk_output4(t4)
        with torch.autograd.profiler.record_function("CONVLAYER5_TRUNKNET"):
          t6 = self.trunk_output5(t5)
        # Reshape to (batch, (nelx+1), (nely+1), 2)
        x1 = x1.view(-1, (nely + 1), (nelx + 1), 2)
        t6 = t6.view(-1, (nely + 1), (nelx + 1), 2)

        # Element-wise multiply
        out = x1 * t6
        out = out + self.b
        return out


def ssim_loss(pred, target, data_range=1.0):
    """
    Computes 1 - SSIM between pred and target.

    pred, target: Tensors of shape [batch_size, H, W, C]
                  In your code, C=2 for 2 displacement channels.
    data_range:   The difference between the maximum and minimum
                  possible values in 'pred' and 'target'.
                  For normalized [0,1] data, data_range=1.0.
                  Adjust if your displacements are in a larger range.
    """
    # permute to: [batch_size, channels, height, width]
    pred_perm = pred.permute(0, 3, 1, 2)
    target_perm = target.permute(0, 3, 1, 2)

    ssim_val = ssim(pred_perm, target_perm, data_range=data_range, size_average=True)
    return 1.0 - ssim_val

def create_deeponet_dataset(branch_input, trunk_input, out_data, history_length=history_length):
    branch_list = []
    trunk_list = []
    output_list = []

    for i in range(history_length, len(branch_input)+1):
        branch_list.append(branch_input[i-history_length : i])
        trunk_list.append(trunk_input[i-1])
        output_list.append(out_data[i-1])

    # Convert entire lists to NumPy arrays first (if they're not already Tensors)
    branch_array = np.array(branch_list)  # shape: (num_samples, 10, 20, 60)
    trunk_array = np.array(trunk_list)    # shape: (num_samples, 21, 61, 2)
    output_array = np.array(output_list)  # shape: (num_samples, 21, 61, 2)

    # Then convert to float Tensors
    branch_data = torch.from_numpy(branch_array).float()
    trunk_data = torch.from_numpy(trunk_array).float()
    output_data = torch.from_numpy(output_array).float()

    return branch_data, trunk_data, output_data

def batch_dataset(batch_size, branch_input, trunk_input, out_data, ratio=0.8):
    # Use out_data here instead of the global S
    us, xs, ss = branch_input, trunk_input, out_data

    train_size = int(len(us)*ratio)
    us_train = us[:train_size]
    xs_train = xs[:train_size]
    ss_train = ss[:train_size]

    us_test = us[train_size:]
    xs_test = xs[train_size:]
    ss_test = ss[train_size:]

    # If they are already Tensors (from create_deeponet_dataset),
    # no need to wrap again. Possibly just cast to float()
    # if your dataset is double or some other type:
    us_train = us_train.float()
    xs_train = xs_train.float()
    ss_train = ss_train.float()

    train_dataset = torch.utils.data.TensorDataset(us_train, xs_train, ss_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False)

    us_test = us_test.float()
    xs_test = xs_test.float()
    ss_test = ss_test.float()

    test_dataset = torch.utils.data.TensorDataset(us_test, xs_test, ss_test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)

    return train_dataloader, test_dataloader
with open('FEA_train_data_'+str(nelx)+'x'+str(nely)+'.pkl', 'rb') as f:
    All_data = pickle.load(f)
# Get training data
Branch_x = np.array([ All_data[i]['x']for i in range(len(All_data))])
Trunk_F = np.array([ All_data[i]['F'].reshape((nely+1),(nelx+1),2) for i in  range(len(All_data))])
Output_S = np.array([ All_data[i]['U'].reshape((nely+1),(nelx+1),2) for i in  range(len(All_data))])
Branch_x.shape, Trunk_F.shape, Output_S.shape
# 1) Create your dataset with the function:
branch_data, trunk_data, output_data = create_deeponet_dataset(Branch_x, Trunk_F, Output_S, history_length=history_length)
print(f"Branch data shape: {branch_data.shape}")
print(f"Trunk data shape: {trunk_data.shape}")
print(f"Output data shape: {output_data.shape}")
# All_changes, All_errors =
model = DeepONet(neurons=40, output_neurons=(nelx + 1) * (nely + 1) * 2)

# Example usage
batch_size = 10
train_dataloader, test_dataloader = batch_dataset(batch_size, branch_data, trunk_data, output_data, ratio=0.8)
%%time
# Define model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepONet().to(device)
criterion = nn.MSELoss()
# criterion = nn.L1Loss()
# criterion = ssim_loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop parameters
epochs = 5000
patience = 500  # Number of epochs to wait for improvement
train_losses = []  # Store the average training loss for each epoch
test_losses = []   # Store the average test loss for each epoch

# Early stopping variables
best_test_loss = float('inf')
epochs_without_improvement = 0
best_model_path = 'CRONet_model_'+str(nelx)+'x'+str(nely)+'.pth'  # Path to save the best model

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for i, (branch_batch, trunk_batch, target_batch) in enumerate(train_dataloader):
        # Move data to device (GPU or CPU)
        branch_batch = branch_batch.to(device)  # shape (batch_size, 10, 20, 60)
        trunk_batch  = trunk_batch.to(device)   # shape (batch_size, 21, 61, 2)
        target_batch = target_batch.to(device)  # shape (batch_size, 21, 61, 2)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass:
        #   The new DeepONet expects forward(branch_input, trunk_input).
        outputs = model(branch_batch, trunk_batch)

        # Compute loss
        loss = criterion(outputs, target_batch)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate running loss
        running_loss += loss.item()

    # Calculate and store the average training loss for the epoch
    average_train_loss = running_loss / len(train_dataloader)
    train_losses.append(average_train_loss)

    # Testing (evaluation) loop
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for branch_batch, trunk_batch, target_batch in test_dataloader:
            branch_batch = branch_batch.to(device)
            trunk_batch  = trunk_batch.to(device)
            target_batch = target_batch.to(device)

            outputs = model(branch_batch, trunk_batch)
            loss = criterion(outputs, target_batch)
            test_loss += loss.item()

    average_test_loss = test_loss / len(test_dataloader)
    test_losses.append(average_test_loss)

    print(f'Epoch [{epoch+1}/{epochs}], '
          f'Train Loss: {average_train_loss:.6f}, '
          f'Test Loss: {average_test_loss:.6f}')

    # Early Stopping checks
    if average_test_loss < best_test_loss:
        best_test_loss = average_test_loss
        epochs_without_improvement = 0
        # Save the best model
        torch.save(model.state_dict(), best_model_path)
        print(f'--> New best model saved with test loss: {best_test_loss:.6f}')
    else:
        epochs_without_improvement += 1
        print(f'--> No improvement for {epochs_without_improvement} epoch(s)')

    if epochs_without_improvement >= patience:
        print(f'Early stopping triggered after {epoch+1} epochs.')
        break

# Load the best model after training
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    print(f'Best model loaded with test loss: {best_test_loss:.6f}')
else:
    print('No best model found, using the last trained model.')

# Visualize the losses
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', color='b', label='Training Loss')
plt.plot(range(1, len(test_losses) + 1), test_losses, marker='o', linestyle='-', color='r', label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Test Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

### Independent test loop
%%time
# Testing loop
model.eval()
test_loss = 0.0
k = 0
with torch.no_grad():
    for us_test, xs_test, ss_test in test_dataloader:
        print(us_test.shape, xs_test.shape, ss_test.shape)

        us_test, xs_test, ss_test = us_test.to(device), xs_test.to(device), ss_test.to(device)
        outputs = model(xs_test, us_test)
        # print(outputs)
        loss = criterion(outputs, ss_test)
        test_loss += loss.item()

average_test_loss = test_loss / len(test_dataloader)
test_losses.append(average_test_loss)
print(average_test_loss)
%%time
# Set the model to evaluation mode
model.eval()
for i, (branch_batch, trunk_batch, target_batch) in enumerate(test_dataloader):
    if i <=100:
        print(f'\n\n ########################## Analysis for the {i}th iteration ##########################')
        # Move data to device (GPU or CPU)
        branch_batch = branch_batch.to(device)  # shape (batch_size, 10, 20, 60)
        trunk_batch  = trunk_batch.to(device)   # shape (batch_size, 21, 61, 2)
        target_batch = target_batch.to(device)  # shape (batch_size, 21, 61, 2)
        with torch.no_grad():
            prediction = model(branch_batch, trunk_batch)  # Output will have shape (1, (nelx+1), (nely+1), 2)

        # Print the prediction (or further process it as needed)
        # print("Prediction Shape:", prediction.shape)
        # Convert to numpy if they are torch tensors
        # if hasattr(target_batch, 'numpy'):
        #     target_batch = target_batch.numpy()
        # if hasattr(prediction, 'numpy'):
        #     prediction = prediction.numpy()
        if isinstance(target_batch, torch.Tensor):
            target_batch = target_batch.detach().cpu().numpy()
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().cpu().numpy()

        # Calculate the mean squared error
        mse = np.mean((target_batch - prediction) ** 2)
        print(mse)


        # Extract the two (nelx+1)x(nely+1) matrices from predicted and true S
        predicted_S1 = prediction[0, :, :, 0]#.cpu().numpy()
        predicted_S2 = prediction[0, :, :, 1]#.cpu().numpy()
        true_S1 = target_batch[0, :, :, 0]#.cpu().numpy()
        true_S2 = target_batch[0, :, :, 1]#.cpu().numpy()

        # Plot the true and predicted matrices
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        im1 = axes[0, 0].imshow(predicted_S1, aspect='auto', cmap='viridis')
        # Plot predicted S1
        axes[0, 0].imshow(predicted_S1, aspect='auto', cmap='viridis')
        axes[0, 0].set_title('Predicted S1')
        axes[0, 0].set_xlabel('Dimension 2')
        axes[0, 0].set_ylabel('Dimension 1')
        fig.colorbar(im1, ax=axes[0, 0])

        # Plot true S1
        im2 = axes[0, 1].imshow(true_S1, aspect='auto', cmap='viridis')
        axes[0, 1].imshow(true_S1, aspect='auto', cmap='viridis')
        axes[0, 1].set_title('True S1')
        axes[0, 1].set_xlabel('Dimension 2')
        axes[0, 1].set_ylabel('Dimension 1')
        fig.colorbar(im2, ax=axes[0, 1])

        # Plot predicted S2
        im3 = axes[1, 0].imshow(predicted_S2, aspect='auto', cmap='viridis')
        axes[1, 0].imshow(predicted_S2, aspect='auto', cmap='viridis')
        axes[1, 0].set_title('Predicted S2')
        axes[1, 0].set_xlabel('Dimension 2')
        axes[1, 0].set_ylabel('Dimension 1')
        fig.colorbar(im3, ax=axes[1, 0])

        # Plot true S2
        im4 = axes[1, 1].imshow(true_S2, aspect='auto', cmap='viridis')
        axes[1, 1].imshow(true_S2, aspect='auto', cmap='viridis')
        axes[1, 1].set_title('True S2')
        axes[1, 1].set_xlabel('Dimension 2')
        axes[1, 1].set_ylabel('Dimension 1')
        fig.colorbar(im4, ax=axes[1, 1])

        plt.tight_layout()
        plt.show()

### Phase 3: Continue the Topology Optimization from the last training batch until convergence
# Get training data
with open('FEA_train_data_'+str(nelx)+'x'+str(nely)+'.pkl', 'rb') as f:
    All_data = pickle.load(f)
Branch_x = np.array([ All_data[i]['x']for i in range(len(All_data))])
Trunk_F = np.array([ All_data[i]['F'].reshape((nely+1),(nelx+1),2) for i in  range(len(All_data))])
Output_S = np.array([ All_data[i]['U'].reshape((nely+1),(nelx+1),2) for i in  range(len(All_data))])
Branch_x.shape, Trunk_F.shape, Output_S.shape

# Load the model from disk
model_path = 'CRONet_model_'+str(nelx)+'x'+str(nely)+'.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    # model.to(device)
    # model.eval()
else:
    raise FileNotFoundError(f"Model file not found: {model_path}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import time

# from zmq import device
gpu_times = []
cpu_times = []


def truncate_below_threshold(array, threshold=0.1):
    """
    Return a new array where all elements with absolute values below the specified threshold are set to zero.

    Parameters:
    array (np.ndarray): The input array of any dimension.
    threshold (float): The threshold value for absolute values below which elements are set to zero.

    Returns:
    np.ndarray: A new array with elements having absolute values below the threshold set to zero.
    """
    # Make a copy of the array to avoid modifying the original
    truncated_array = np.copy(array)

    # Apply threshold based on absolute values
    truncated_array[np.abs(truncated_array) < threshold] = 0

    return truncated_array

def deep_FE(branch_input, trunk_input, method="gpu"):
    model.eval()
    prediction = None
    elapsed_ms = None
    with torch.no_grad():
        if method == "gpu":
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            prediction = model(branch_input, trunk_input)
            ender.record()
            torch.cuda.synchronize()
            elapsed_ms = starter.elapsed_time(ender)
            gpu_times.append(elapsed_ms)
            print(f"[deep_FE] GPU inference time: {elapsed_ms:.3f} ms")
        elif method == "cpu":
            start_time = time.time()
            prediction = model(branch_input, trunk_input)
            torch.cuda.synchronize() if branch_input.is_cuda else None
            elapsed_ms = (time.time() - start_time) * 1000.0
            cpu_times.append(elapsed_ms)
            print(f"[deep_FE] CPU (wall-clock) inference time: {elapsed_ms:.3f} ms")
        else:
            raise ValueError("method must be 'gpu' or 'cpu'")

    # Move result back to CPU numpy
    prediction = prediction.cpu().numpy()
    print(f"Prediction shape {prediction.shape}")
    return prediction.reshape(-1, 1).round()

def create_deeponet_dataset(branch_input, trunk_input, out_data, history_length=history_length):
    # print(f'branch_input_type = {branch_input},\ntrunk_input_type = {trunk_input},\noutput_data_type = {out_data}')
    branch_list = []
    trunk_list = []
    output_list = []

    # for i in range(history_length, len(branch_input)+1):
    #     branch_list.append(branch_input[i-history_length : i])
    #     trunk_list.append(trunk_input[i-1])
    #     output_list.append(out_data[i-1])
    
    branch_list.append(branch_input[-history_length:])
    trunk_list.append(trunk_input[-1])
    output_list.append(out_data[-1])

    # branch_list = branch_list[:1]
    # trunk_list = trunk_list[:1]
    # output_list = output_list[:1]
    # print(len(branch_list))
    # print(len(trunk_list))
    # print(len(output_list))
    # Convert entire lists to NumPy arrays first (if they're not already Tensors)
    branch_array = np.array(branch_list)  # shape: (num_samples, 10, 20, 60)
    trunk_array = np.array(trunk_list)    # shape: (num_samples, 21, 61, 2)
    output_array = np.array(output_list)  # shape: (num_samples, 21, 61, 2)

    # Then convert to float Tensors
    branch_data = torch.from_numpy(branch_array).float()
    trunk_data = torch.from_numpy(trunk_array).float()
    output_data = torch.from_numpy(output_array).float()

    return branch_data, trunk_data, output_data
def top_deepOnet_mp(nelx, nely, volfrac, penal, rmin, method="gpu"):

    F = np.zeros((2 * (nelx + 1) * (nely + 1), 1))
    F[1, 0] = -1

    # INITIALIZE
    # x = np.ones((nely, nelx)) * volfrac
    loop = 0
    change = 1.0
    fem_cnt = 0
    don_cnt = 0
    U_DON = []
    U_FE = []
    Branch_x_data = list(Branch_x[-10:])
    Trunk_F_data = list(Trunk_F[-10:])
    x = Branch_x_data[-1]
    error  = 0.2

    # START ITERATION
    while change > 0.01 and loop<100:
        xold = x.copy()

        if error >= 0.2:
            print(f'used FEM')
            U = FE(nelx, nely, x, penal)
            fem_cnt += 1
        else:
            print(f'used deep_FE')
            bd, td, _ = create_deeponet_dataset(np.array(Branch_x_data), np.array(Trunk_F_data), np.array(Trunk_F_data), history_length=history_length)
            print(f'bd shape: {bd.shape}, td shape: {td.shape}')
            bd = torch.tensor(bd, dtype=torch.float32).to(device)
            td = torch.tensor(td, dtype=torch.float32).to(device)
            # while(True):
            U = deep_FE(bd, td, method=method)
            # print(U.shape)
            don_cnt +=1
            U_DON.append(U)
        loop += 1

        # OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
        KE = lk()
        c = 0.0
        dc = np.zeros((nely, nelx))
        for ely in range(nely):
            for elx in range(nelx):
                n1 = (nely + 1) * (elx) + ely
                n2 = (nely + 1) * (elx + 1) + ely
                Ue = U[[2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n2 + 2, 2 * n2 + 3, 2 * n1 + 2, 2 * n1 + 3], 0]
                c += x[ely, elx] ** penal * Ue @ KE @ Ue
                dc[ely, elx] = -penal * x[ely, elx] ** (penal - 1) * Ue @ KE @ Ue

        # FILTERING OF SENSITIVITIES
        dc = check(nelx, nely, rmin, x, dc)

        # DESIGN UPDATE BY THE OPTIMALITY CRITERIA METHOD
        x = OC(nelx, nely, x, volfrac, dc)

        Branch_x_data.append(x.copy())
        Trunk_F_data.append(F.reshape((nely+1),(nelx+1),2))

        # PRINT RESULTS
        error = np.mean((x-xold)**2)
        error = np.max(np.abs(x - xold))
        change = np.max(np.abs(x - xold))
        print(f' It.: {loop} Obj.: {c:.4f} Vol.: {np.mean(x):.3f} ch.: {change:.3f}')

        # PLOT DENSITIES
        plt.figure(figsize=(2, 1), dpi=100)  # Reduced figure size and increased dpi for faster plotting
        plt.imshow(-x, cmap='gray', interpolation='none')
        plt.axis('off')  # Disable axis for faster plotting
        # plt.pause(1e-6)  # Keep a small pause to update the plot quickly
        plt.show()  # Close the figure to free memory and improve performance
    print(f'\n Used FEM {fem_cnt} times and DON_FEM {don_cnt} times until convergence')
    # return U_DON, U_FE



# OPTIMALITY CRITERIA UPDATE
def OC(nelx, nely, x, volfrac, dc):
    l1, l2, move = 0, 100000, 0.2
    eps = 1e-12
    # print(np.max(dc))
    while (l2 - l1) > 1e-4:
        lmid = 0.5 * (l2 + l1)
        xnew = np.maximum(0.001, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(np.maximum(-dc / lmid, eps))))))
        if np.sum(xnew) - volfrac * nelx * nely > 0:
            l1 = lmid
        else:
            l2 = lmid
    return xnew

# MESH-INDEPENDENCY FILTER
def check(nelx, nely, rmin, x, dc):
    dcn = np.zeros_like(dc)
    for i in range(nelx):
        for j in range(nely):
            sum_val = 0.0
            for k in range(max(i - round(rmin), 0), min(i + round(rmin), nelx)):
                for l in range(max(j - round(rmin), 0), min(j + round(rmin), nely)):
                    fac = rmin - np.sqrt((i - k) ** 2 + (j - l) ** 2)
                    sum_val += max(0.0, fac)
                    dcn[j, i] += max(0.0, fac) * x[l, k] * dc[l, k]
            dcn[j, i] /= x[j, i] * sum_val
    return dcn

# FE-ANALYSIS
def FE(nelx, nely, x, penal):
    KE = lk()
    K = lil_matrix((2 * (nelx + 1) * (nely + 1), 2 * (nelx + 1) * (nely + 1)))
    F = np.zeros((2 * (nelx + 1) * (nely + 1), 1))
    U = np.zeros((2 * (nelx + 1) * (nely + 1), 1))
    for elx in range(nelx):
        for ely in range(nely):
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edof = np.array([2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n2 + 2, 2 * n2 + 3, 2 * n1 + 2, 2 * n1 + 3])
            K[np.ix_(edof, edof)] += x[ely, elx] ** penal * KE
    # DEFINE LOADS AND SUPPORTS (HALF MBB-BEAM)
    F[1, 0] = -1
    fixeddofs = np.union1d(np.arange(0, 2 * (nely + 1), 2), np.array([2 * (nelx + 1) * (nely + 1) - 1]))
    alldofs = np.arange(2 * (nelx + 1) * (nely + 1))
    freedofs = np.setdiff1d(alldofs, fixeddofs)
    # SOLVE
    U[freedofs, 0] = spsolve(K[freedofs, :][:, freedofs], F[freedofs, 0])
    U[fixeddofs, 0] = 0
    return U #, K, F, KE

# ELEMENT STIFFNESS MATRIX
def lk():
    E = 1.0
    nu = 0.3
    k = np.array([
        1/2 - nu / 6, 1/8 + nu / 8, -1/4 - nu / 12, -1/8 + 3 * nu / 8,
        -1/4 + nu / 12, -1/8 - nu / 8, nu / 6, 1/8 - 3 * nu / 8
    ])
    KE = E / (1 - nu ** 2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
    ])
    return KE

# All_changes, All_errors =
x = torch.randn((1, 1), requires_grad=True)
with torch.autograd.profiler.profile(use_cuda=True) as prof:
  top_deepOnet_mp(nelx, nely, volfrac, penal, rmin, method="cpu")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
# All_changes, All_errors =
top_deepOnet_mp(nelx, nely, volfrac, penal, rmin, method="cpu")
if gpu_times:
    print(f"[deep_FE] Total GPU time: {sum(gpu_times):.3f} ms")
    print(f"[deep_FE] Average GPU time: {np.mean(gpu_times):.3f} ms over {len(gpu_times)} calls")
else:
    print("[deep_FE] No GPU times recorded.")
if cpu_times:
    print(f"[deep_FE] Total CPU time: {sum(cpu_times):.3f} ms")
    print(f"[deep_FE] Average CPU time: {np.mean(cpu_times):.3f} ms over {len(cpu_times)} calls")
else:
    print("[deep_FE] No CPU times recorded.")
### Inference loop for time measurement using BS 1. Use this same loop for power measurement. Run it for long enough duration
# iterations
N = 1000
cpu_times = []
Branch_x_data = list(Branch_x[-10:])
Trunk_F_data = list(Trunk_F[-10:])
x = Branch_x_data[-1]
print(f'Deep_FE')
bd, td, _ = create_deeponet_dataset(np.array(Branch_x_data), np.array(Trunk_F_data), np.array(Trunk_F_data), history_length=history_length)
print(f'bd shape: {bd.shape}, td shape: {td.shape}')
bd = torch.tensor(bd, dtype=torch.float32).to(device)
td = torch.tensor(td, dtype=torch.float32).to(device)

# Here CPU does not mean its running on CPU. It means the measurement is done using wall clock timer instead of GPU events. 
# The device var defines where the model and data are located (GPU or CPU). 
# So if device is GPU, the model will run on GPU but the timing will be done using wall clock time. If device is CPU, then both the model and timing will be on CPU.
for i in range(N):
    U = deep_FE(bd, td, method='cpu')
if cpu_times:
    print(f"[deep_FE] Total CPU time: {sum(cpu_times):.3f} ms")
    print(f"[deep_FE] Average CPU time: {np.mean(cpu_times):.3f} ms over {len(cpu_times)} calls")
else:
    print("[deep_FE] No CPU times recorded.")