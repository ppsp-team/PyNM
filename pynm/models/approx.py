import math
import torch
import gpytorch
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

class GPModel(ApproximateGP):
    """ Class for GPyTorch model.

    Attributes
    ----------
    mean_module : gpytorch Mean
        Module to calculate mean.
    covar_module : gpytorch Kernel
        Module to calculate covariance.
    """
    def __init__(self, inducing_points,nu=2.5,length_scale=1,length_scale_bounds=(1e-5,1e5)):
        """ Create a GPModel object.

        Parameters
        ----------
        inducing_points: array
            Array of inducing points.
        length_scale: float, default=1
            Length scale parameter of Matern kernel.
        length_scale_bounds: pair of floats >= 0 or 'fixed', default=(1e-5, 1e5)
            The lower and upper bound on length_scale. If set to 'fixed', ‘length_scale’ cannot be changed during hyperparameter tuning.
        nu: float, default=2.5
            Nu parameter of Matern kernel.
        
        Raises
        ------
        ValueError
            Invalid argument for length_scale_bounds
        """
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()

        if length_scale_bounds == 'fixed':
            constraint = gpytorch.constraints.Interval(length_scale - 0.001,length_scale + 0.0001)
        elif isinstance(length_scale_bounds,tuple):
            constraint = gpytorch.constraints.Interval(length_scale_bounds[0],length_scale_bounds[1])
        else:
            raise ValueError('Invalid argument for length_scale_bounds.')
        prior = gpytorch.priors.NormalPrior(length_scale,1)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=nu,lengthscale_prior=prior),lengthscale_contraint = constraint)

    def forward(self, x):
        """ Calculate forward pass of GPModel.

        Parameters
        ----------
        x: Tensor
            Data tensor.
        
        Returns
        -------
        MultivariateNormal object
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SVGP:
    """ Class for SVGP model.

    Attributes
    ----------
    train_loader: pytorch DataLoader
        DataLoader for training data.
    test_loader: pytorch DataLoader
        DataLoader for test data.
    inducing_points: array
        Subset of training data to use as inducing points.
    n_train: int
        Number of training points.
    n_test: int
        Number of test points.
    model: GPModel
        Instance of GPModel class.
    likelihood: gpytorch Likelihood
        Gaussian likelihood function.
    loss: list
        Loss for each epoch of training.
    """
    def __init__(self,X_train,X_test,y_train,y_test,n_inducing=500,batch_size=256,nu=2.5,length_scale=1,length_scale_bounds=(1e-5,1e5)):
        """ Create a SVGP object.

        Parameters
        ----------
        X_train: array
            Training confounds with categorical values dummy encoded.
        X_test: array
            Test confounds with categorical values dummy encoded.
        y_train: array
            Training score/response variable.
        y_test: array
            Test score/response variable.
        length_scale: float, default=1
            Length scale parameter of Matern kernel.
        length_scale_bounds: pair of floats >= 0 or 'fixed', default=(1e-5, 1e5)
            The lower and upper bound on length_scale. If set to 'fixed', ‘length_scale’ cannot be changed during hyperparameter tuning.
        nu: float, default=2.5
            Nu parameter of Matern kernel.
        batch_size: int, default=256
            Batch size for SVGP model training and prediction.
        n_inducing: int, default=500
            Number of inducing points for SVGP model.
        """
        # Get data in torch format
        train_x = torch.from_numpy(X_train).contiguous()
        test_x = torch.from_numpy(X_test).double().contiguous()
        train_y = torch.from_numpy(y_train).contiguous()
        test_y = torch.from_numpy(y_test).double().contiguous()

        # Create datasets
        train_dataset = TensorDataset(train_x, train_y)
        test_dataset = TensorDataset(test_x, test_y)

        # Create dataloaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        inducing_idx = np.random.choice(np.array(range(train_x.shape[0])),size=n_inducing)
        self.inducing_points = train_x[inducing_idx, :]
        self.n_train = train_y.size(0)
        self.n_test = test_y.size(0)

        self.model = GPModel(inducing_points=self.inducing_points,nu=nu,length_scale=length_scale,length_scale_bounds=length_scale_bounds).double()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.initialize(noise=torch.std(train_x))

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
        
        self.loss = []
    
    def train(self,num_epochs=20):
        """ Trains the SVGP model.

        Parameters
        ----------
        num_epochs: int
            Number of epochs (full passes through dataset) to train for.
        """
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam([{'params': self.model.parameters()},{'params': self.likelihood.parameters()}], lr=0.01)

        # Loss object. We're using the VariationalELBO
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=self.n_train)

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
        """ Predict from SVGP model.

        Returns
        ----------
        array
            Model predictions (mean of predictive distribution).
        array
            Model uncertainty (standard deviation of predictive distribution).
        """
        self.model.eval()
        self.likelihood.eval()

        mean = torch.tensor([0.])
        sigma = torch.tensor([0.])
        with torch.no_grad():
            for x_batch, y_batch in self.test_loader:
                preds = self.likelihood(self.model(x_batch)) # get likelihood variance + posterior GP variance
                mean = torch.cat([mean, preds.mean.cpu()])
                sigma = torch.cat([sigma, torch.sqrt(preds.variance.cpu())])
        mean = mean[1:]
        sigma = sigma[1:]
        return mean, sigma