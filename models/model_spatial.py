import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

class NN_Encoder(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(NN_Encoder, self).__init__()
        self.enc1 = nn.Linear(inp_dim, 64)
        self.mu = nn.Linear(64, out_dim)
        self.logvar = nn.Linear(64, out_dim)

    def forward(self, x):
        x = F.relu(self.enc1(x))
        return self.mu(x), self.logvar(x)

class NN_Decoder(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(NN_Decoder, self).__init__()
        self.dec1 = nn.Linear(inp_dim, 64)
        self.dec2 = nn.Linear(64, out_dim)

    def forward(self, x):
        x = F.relu(self.dec1(x))
        x = self.dec2(x)
        return x

class NN_embedding(nn.Module):
    def __init__(self, params, grid_dim, seq_len):
        super(NN_embedding, self).__init__()
        self.params = params
        self.grid_dim = grid_dim
        self.embedding_dim = self.params.spatial_embedding_dim
        self.seq_len = seq_len
        self.inp_dim = grid_dim[0] * grid_dim[1]

        self.Encoder = NN_Encoder (
            inp_dim = self.inp_dim,
            out_dim = self.embedding_dim
        )
        self.Decoder = NN_Decoder (
            inp_dim = self.embedding_dim,
            out_dim = self.inp_dim
        )
        self.MSELoss = nn.MSELoss()

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + (eps * std)

    def KL_divergence(self, mu, logvar):
        loss = -0.5 * torch.sum(logvar - mu.pow(2) - torch.exp(logvar) + 1)
        return loss

    def decode(self, z, out_len=None):
        if out_len is None:
            out_len = self.seq_len
        y = self.Decoder(z.contiguous().view(-1, self.embedding_dim))
        y = y.view(-1, self.grid_dim[0], self.grid_dim[1]).contiguous().\
                    view(-1, out_len, self.grid_dim[0], self.grid_dim[1])
        return y

    def encode(self, x, loss=False):
        x1 = x.view(-1, self.grid_dim[0], self.grid_dim[1]).contiguous().view(-1, self.inp_dim)
        mu, logvar = self.Encoder(x1)
        if self.params.use_VAE == True:
            z = self.reparametrize(mu, logvar)
            var_loss = self.KL_divergence(mu, logvar)
        else:
            z = mu
            var_loss = torch.tensor(0.0)

        if loss == True:
            return z, var_loss
        else:
            return z

    def output(self, x):
        out_len = x.shape[1]
        return self.decode(self.encode(x), out_len)

    def forward(self, x):
        seq_len = x.shape[1]
        z, var_loss = self.encode(x, loss=True)
        y = self.decode(z, seq_len)

        rec_loss = self.MSELoss(x, y)
        z = z.view(-1, seq_len, z.shape[1])

        return {
            'emb': z,
            'rec_loss': rec_loss,
            'var_loss': var_loss
        }


class PCA_embedding_scikit(nn.Module):
    def __init__(self, feature, n_modes):
        super(PCA_embedding_scikit, self).__init__()
        self.epsilon = 1e-14
        self.grid_dim = feature.shape[-2:]
        self.pca_pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=n_modes))])
        X = feature.reshape(-1,self.grid_dim[0]*self.grid_dim[1])
        Y = self.pca_pipeline.fit_transform(X)
        self.mean, self.std = Y.mean(axis=0), Y.std(axis=0)

    def inverse(self, y):
        shape = y.shape
        y = y.view(-1, shape[2])
        y = y * self.std + self.mean
        x = self.pca_pipeline.inverse_transform(y.numpy())
        x = x.reshape(shape[0], shape[1], -1).reshape(shape[0], shape[1], self.grid_dim[0], self.grid_dim[1])
        return torch.tensor(x)

    def forward(self, x):
        shape = x.shape
        x = x.view(-1,shape[2], shape[3]).contiguous().view(-1, shape[2]*shape[3])
        y = self.pca_pipeline.transform(x.numpy())
        y = (y - self.mean) / (self.std + self.epsilon)
        y = y.reshape(shape[0], shape[1], -1)
        return torch.tensor(y)


class PCA_embedding(nn.Module):
    def __init__(self, feature, n_modes):
        super(PCA_embedding, self).__init__()
        self.epsilon = 1e-14
        self.grid_dim = feature.shape[-2:]
        self.n_modes = n_modes
        X = torch.from_numpy(feature)
        X = X.view(-1,self.grid_dim[0]*self.grid_dim[1])
        self.X_mean, self.X_std = X.mean(dim=0), X.std(dim=0)
        self._fit_transform(X)

    def _fit_transform(self, X):
        X = (X - self.X_mean) / (self.X_std + self.epsilon)
        U,S,V = torch.svd(X)
        self.eigvecs=V[:,:self.n_modes]
        Y = torch.mm(X,self.eigvecs)
        self.eigvals = S
        self.Y_mean, self.Y_std = Y.mean(dim=0), Y.std(dim=0)
        self.explain_variance()
        return Y

    def _transform(self, X):
        X = (X - self.X_mean) / (self.X_std + self.epsilon)
        Y = torch.mm(X,self.eigvecs)
        Y = (Y - self.Y_mean) / (self.Y_std + self.epsilon)
        return Y

    def forward(self, x):
        shape = x.shape
        x = x.view(-1, shape[2], shape[3]).contiguous().view(-1, shape[2]*shape[3])
        y = self._transform(x)
        y = y.view(shape[0], shape[1], -1).contiguous()
        return y

    def inverse(self, y):
        shape = y.shape
        y = y.contiguous().view(-1, shape[2])
        y = y * self.Y_std + self.Y_mean
        x = torch.mm(y, self.eigvecs.t())
        x = x * self.X_std + self.X_mean
        x = x.view(shape[0], shape[1], -1).contiguous().view(shape[0], shape[1], self.grid_dim[0], self.grid_dim[1])
        return x
    
    def explain_variance(self):
        S = self.eigvals
        self.explained_variance_ratio_ = (S*S)[:self.n_modes] / (S*S).sum()
        self.explained_variance_ = torch.sum(self.explained_variance_ratio_)