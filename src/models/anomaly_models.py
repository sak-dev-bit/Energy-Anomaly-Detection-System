from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import torch
import torch.nn as nn

class IsolationForestModel:
    def __init__(self, n_estimators=100, contamination=0.05):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42
        )

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        preds = self.model.predict(X)
        return (preds == -1).astype(int)

class LOFModel:
    def __init__(self, n_neighbors=20, contamination=0.05):
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True
        )

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        preds = self.model.predict(X)
        return (preds == -1).astype(int)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

class AutoencoderModel:
    def __init__(self, input_dim, latent_dim=8, lr=1e-3):
        self.model = Autoencoder(input_dim, latent_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def fit(self, X, epochs=20, batch_size=32):
        X_tensor = torch.tensor(X, dtype=torch.float32)

        for epoch in range(epochs):
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]

                recon = self.model(batch)
                loss = self.criterion(recon, batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict(self, X, threshold=None):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        recon = self.model(X_tensor)

        error = torch.mean((X_tensor - recon)**2, dim=1).detach().numpy()

        if threshold is None:
            threshold = error.mean() + 2 * error.std()

        return (error > threshold).astype(int), error
