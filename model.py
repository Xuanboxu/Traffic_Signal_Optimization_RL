import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary


class TrainModel:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._model = self._build_model(num_layers, width)
        self._loss_fn = nn.MSELoss()  # Mean Squared Error loss
        self._optimizer = optim.Adam(self._model.parameters(), lr=self._learning_rate)

    def _build_model(self, num_layers, width):
        """
        Build a fully connected deep neural network.
        """
        layers = [nn.Linear(self._input_dim, width), nn.ReLU()]
        for _ in range(num_layers):
            layers.extend([nn.Linear(width, width), nn.ReLU()])
        layers.append(nn.Linear(width, self._output_dim))
        return nn.Sequential(*layers)

    def predict_one(self, state):
        """
        Predict the action values from a single state.
        """
        self._model.eval()  # Set the model to evaluation mode
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            return self._model(state_tensor).squeeze(0).numpy()

    def predict_batch(self, states):
        """
        Predict the action values from a batch of states.
        """
        self._model.eval()  # Set the model to evaluation mode
        state_tensor = torch.tensor(states, dtype=torch.float32)
        with torch.no_grad():
            return self._model(state_tensor).numpy()

    def train_batch(self, states, q_sa):
        """
        Train the neural network using the updated q-values.
        """
        self._model.train()  # Set the model to training mode
        state_tensor = torch.tensor(states, dtype=torch.float32)
        q_sa_tensor = torch.tensor(q_sa, dtype=torch.float32)

        self._optimizer.zero_grad()  # Clear gradients
        predictions = self._model(state_tensor)
        loss = self._loss_fn(predictions, q_sa_tensor)
        loss.backward()  # Backpropagate
        self._optimizer.step()  # Update weights

    def save_model(self, path):
        """
        Save the current model and its architecture summary.
        """
        os.makedirs(path, exist_ok=True)
        model_file = os.path.join(path, 'trained_model.pth')
        torch.save(self._model.state_dict(), model_file)
        # Save the model architecture summary
        with open(os.path.join(path, 'model_structure.txt'), 'w') as f:
            f.write(str(summary(self._model, (self._input_dim,))))

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def batch_size(self):
        return self._batch_size


class TestModel:
    def __init__(self, input_dim, model_path):
        self._input_dim = input_dim
        self._model = self._load_my_model(model_path)

    def _load_my_model(self, model_folder_path):
        """
        Load the model stored in the specified folder.
        """
        model_file_path = os.path.join(model_folder_path, 'trained_model.pth')
        
        if os.path.isfile(model_file_path):
            model = self._build_model()
            model.load_state_dict(torch.load(model_file_path))
            model.eval()  # Set the model to evaluation mode
            return model
        else:
            raise FileNotFoundError("Model not found at the specified path.")

    def _build_model(self):
        """
        Rebuild the architecture for loading the model.
        """
        layers = [nn.Linear(self._input_dim, 64), nn.ReLU()]
        layers.extend([nn.Linear(64, 64), nn.ReLU()])  # Example layers (customize as needed)
        layers.append(nn.Linear(64, self._input_dim))  # Output layer
        return nn.Sequential(*layers)

    def predict_one(self, state):
        """
        Predict the action values from a single state.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            return self._model(state_tensor).squeeze(0).numpy()

    @property
    def input_dim(self):
        return self._input_dim
