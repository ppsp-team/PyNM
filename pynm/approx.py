import math
import torch
import gpytorch
import statsmodels.api as sm
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SVGP:
    def __init__(self,conf_mat,score,ctr_mask,n_inducing=500,batch_size=256):
        # Get data in torch format
        X = torch.from_numpy(conf_mat)
        y = torch.from_numpy(score)

        # Split into train/test
        train_x = X[ctr_mask].contiguous()
        train_y = y[ctr_mask].contiguous()
        test_x = X.double().contiguous()
        test_y = y.double().contiguous()

        # Create datasets
        train_dataset = TensorDataset(train_x, train_y)
        test_dataset = TensorDataset(test_x, test_y)

        # Create dataloaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.inducing_points = train_x[:n_inducing, :]
        self.n_train = train_y.size(0)
        self.n_test = test_y.size(0)

        self.model = GPModel(inducing_points=self.inducing_points).double()
        #model = model.double()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
    
    def train(self,num_epochs=10):
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam([{'params': self.model.parameters()},{'params': self.likelihood.parameters()}], lr=0.01)

        # Loss object. We're using the VariationalELBO
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=self.n_train)

        self.loss = []
        epochs_iter = tqdm(range(num_epochs), desc="Epoch")
        for i in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
            minibatch_iter = tqdm(self.train_loader, desc="Minibatch", leave=False)
            for x_batch, y_batch in minibatch_iter:
                optimizer.zero_grad()
                output = self.model(x_batch)
                loss = -mll(output, y_batch)
                minibatch_iter.set_postfix(loss=loss.item())
                loss.backward()
                optimizer.step()
            self.loss.append(loss.item())
    
    def predict(self):
        self.model.eval()
        self.likelihood.eval()

        means = torch.tensor([0.])
        sigmas = torch.tensor([0.])
        with torch.no_grad():
            for x_batch, y_batch in self.test_loader:
                preds = self.model(x_batch)
                means = torch.cat([means, preds.mean.cpu()])
                sigmas = torch.cat([sigmas, preds.variance.cpu()])
        means = means[1:]
        sigmas = sigmas[1:]
        return means, sigmas