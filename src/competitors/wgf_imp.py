import numpy as np
import torch
import logging
from torch.utils.data import DataLoader
import time
import math

def MAE(X_filled, X_true, mask):
    """Mean Absolute Error for missing values"""
    return np.abs(X_filled[mask == 1] - X_true[mask == 1]).mean()

def nanmean(X, axis=0):
    X_np = X.detach().cpu().numpy()
    return torch.tensor(np.nanmean(X_np, axis=axis), dtype=X.dtype)

def parzen_window_pdf(x, data, bandwidth=1.0):
    n, d = data.shape
    diff = (x[None, :] - data) / bandwidth
    norm = np.exp(-0.5 * np.sum(diff ** 2, axis=1))
    return np.sum(norm) / (n * (bandwidth ** d) * np.sqrt(2 * np.pi) ** d)

class Trainer:
    def __init__(self, model, loss_type="dsm", device=None):
        self.model = model
        self.loss_type = loss_type
        self.device = device if device is not None else torch.device("cpu")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
    def train_step(self, data):
        self.model.train()
        x = data[0].to(self.device) if isinstance(data, (list, tuple)) else data.to(self.device)
        x = x.float()
        self.optimizer.zero_grad()

        out = self.model(x)
        loss = ((out - x) ** 2).mean()
        loss.backward()
        self.optimizer.step()
        return loss

class ToyMLP(torch.nn.Module):
    def __init__(self, input_dim, units=[256, 256]):
        super().__init__()
        layers = []
        prev = input_dim
        for u in units:
            layers.append(torch.nn.Linear(prev, u))
            layers.append(torch.nn.ReLU())
            prev = u
        layers.append(torch.nn.Linear(prev, input_dim))
        self.net = torch.nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class Energy(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x):
        return self.net(x)
    def functorch_score(self, x):
        x = x.clone().detach().requires_grad_(True)
        y = self.forward(x)
        grad = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
        return grad

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return self.data[idx]

def train_step(data, trainer):
    return trainer.train_step(data)

"""
# Credit to https://github.com/Ending2015a/toy_gradlogp 
"""


def xRBF(sigma=-1):
    bandwidth = sigma

    def compute_rbf(X, Y):

        XX = torch.matmul(X, X.t())
        XY = torch.matmul(X, Y.t())
        YY = torch.matmul(Y, Y.t())

        dnorm2 = -2 * XY + torch.diag(XX).unsqueeze(1) + torch.diag(YY).unsqueeze(0)
        if bandwidth < 0:
            median = torch.quantile(dnorm2.detach(), q=0.5) / (2 * math.log(X.shape[0] + 1))
            sigma = torch.sqrt(median)
        else:
            sigma = bandwidth
        gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        K_XY = torch.exp(-gamma * dnorm2)

        dx_K_XY = -torch.matmul(K_XY, X)
        sum_K_XY = torch.sum(K_XY, dim=1)
        for i in range(X.shape[1]):
            dx_K_XY[:, i] = dx_K_XY[:, i] + torch.multiply(X[:, i], sum_K_XY)
        dx_K_XY = dx_K_XY / (1.0e-8 + sigma ** 2)

        return dx_K_XY, K_XY

    return compute_rbf

class NeuralGradFlowImputer(object):
    def __init__(self, initializer=None, entropy_reg=10.0, eps=0.01, lr=1.0e-1,
                 opt=torch.optim.Adam, niter=50, kernel_func=xRBF,
                 mlp_hidden=[256, 256], score_net_epoch=2000, score_net_lr=1.0e-3, score_loss_type="dsm",
                 log_pdf=parzen_window_pdf, bandwidth=10.0, sampling_step=500, log_path="./neuralGFImpute",
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 batchsize=128, n_pairs=1, noise=0.1, scaling=.9):
        super(NeuralGradFlowImputer, self).__init__()
        self.eps = eps
        self.lr = lr
        self.opt = opt
        self.niter = niter
        self.batchsize = batchsize
        self.n_pairs = n_pairs
        self.noise = noise
        self.scaling = scaling
        self.sampling_step = sampling_step
        self.device = device
        self.mlp_hidden = mlp_hidden
        self.score_net_epoch = score_net_epoch
        self.score_loss_type = score_loss_type
        self.score_net_lr = score_net_lr
        self.log_path = log_path
        self.initializer = initializer
        self.entropy_reg = entropy_reg

        # if os.path.exists(os.path.join("./", self.log_path)):
        #     shutil.rmtree(os.path.join("./", self.log_path))
        # self.writer = score_tuple.logWriter(os.path.join("./", self.log_path))
        self.bandwidth = bandwidth



        # kernel func and concerning grad
        self.kernel_func = kernel_func(self.bandwidth)
        self.grad_val_kernel = self.kernel_func

        # log pdf func and concerning grad



        # self.sk = SamplesLoss("sinkhorn", p=2, blur=eps, scaling=scaling, backend="tensorized")
        # self.data_step = SVGDScoreFunction(score_func=self.log_pdf, kernel_func=rbf_kernel(1.0))


    def _sum_kernel(self, X, Y):
        """
        wrap the kernel function to obtain grad_and_values
        :return: scalar, kernel value
        """
        K_XY = self.kernel_func(X, Y)
        return torch.sum(K_XY), K_XY

    def knew_imp_sampling(self, data, bandwidth, data_number, score_func, grad_optim, iter_steps, mask_matrix, clamp=False):
        """
        svgd sampling function
        :param data:
        :param data_number:
        :param score_func:
        :param grad_optim:
        :param iter_steps:
        :param mask_matrix:
        :return:
        """
        for _ in range(iter_steps):
            # with torch.no_grad():
            eval_score = score_func(data)
            eval_grad_k, eval_value_k = self.grad_val_kernel(data, data)
            eval_score = eval_score.detach()
            eval_grad_k = eval_grad_k.detach()
            eval_value_k = eval_value_k.detach()

            # svgd gradient
            grad_tensor = -1.0 * (torch.matmul(eval_value_k, eval_score) - self.entropy_reg * eval_grad_k) / data_number
            # grad_tensor = -1.0 * eval_score
            if torch.isnan(grad_tensor).any() or torch.isinf(grad_tensor).any():
                ### Catch numerical errors/overflows (should not happen)
                logging.info("Nan or inf loss")
                break
            # mask the corresponding values
            grad_optim.zero_grad()
            data.grad = torch.masked_fill(input=grad_tensor, mask=mask_matrix, value=0.0)
            grad_optim.step()
        
            data = torch.clamp(data, min=0.0, max=1.0).detach() if clamp else data
        return data.detach().clone()

    def train_score_net(self, train_dataloader, outer_loop, score_trainer):
        """
        score network training function
        :param train_dataloader:
        :param outer_loop:
        :param score_trainer:
        :return:
        """
        for e in range(self.score_net_epoch):
            total_loss = 0.0
            for _, data in enumerate(train_dataloader):
                loss = train_step(data, score_trainer)
                total_loss = total_loss + loss.item()


    def fit_transform(self, X, verbose=True, report_interval=10, X_true=None, OTLIM=5000, clamp=False):
        X, X_true = torch.tensor(X), torch.tensor(X_true)
        X = X.clone()
        n, d = X.shape

        # define the score network structure and corresponding trainer
        if d == 3 * 32 * 32 or d == 3 * 64 * 64:
            from src.cnnrgb import CNNNet2
            self.mlp_model = CNNNet2(d).to(self.device)
        else:
            self.mlp_model = ToyMLP(input_dim=d, units=self.mlp_hidden).to(self.device)
        
        self.score_net = Energy(net=self.mlp_model).to(self.device)

        if self.batchsize > n // 2:
            e = int(np.log2(n // 2))
            self.batchsize = 2 ** e
            if verbose:
                logging.info(f"Batchsize larger that half size = {len(X) // 2}. Setting batchsize to {self.batchsize}.")


        mask = torch.isnan(X).float()

        if self.initializer is not None:
            imps = self.initializer.fit_transform(X)
            imps = torch.tensor(imps).float().to(self.device)
            imps = (self.noise * torch.randn(mask.shape, device=self.device).float() + imps)[mask]
        else:
            imps = (self.noise * torch.randn(mask.shape).float() + (1) * nanmean(X, 0))[mask.bool()]
        grad_mask = ~mask.bool()
        X_filled = X.detach().clone()
        X_filled[mask.bool()] = imps

        X_filled = X_filled.to(self.device)
        X_filled.requires_grad_()
        grad_mask = grad_mask.to(self.device)

        optimizer = self.opt([X_filled], lr=self.lr)

        if verbose:
            logging.info(f"batchsize = {self.batchsize}, epsilon = {self.eps:.4f}")

        if X_true is not None:
            maes = np.zeros(self.niter)
            result_list = []

        for i in range(self.niter):
            print(f"Iteration {i + 1}:")

            # trian the score network
            score_trainer = Trainer(model=self.score_net, loss_type=self.score_loss_type, device =self.device)
            train_start_time = time.time()
            train_dataset = MyDataset(data=X_filled)

            train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=self.batchsize)
            self.train_score_net(train_dataloader, i, score_trainer)
            train_end_time = time.time()

            model_start_time = time.time()
            # fill the dataset with SVGD
            # sampling 500 at a time 
            # X_temp = X_filled.detach().clone()
            # for sample_count in range(0, n, 500):
            #     print(".", end=" ")
            #     # print(f"Sample count: {sample_count}")
            #     if sample_count + 500 > n:
            #         sample_count = n - 500
            #     X_temp[sample_count:sample_count + 500, :] = self.knew_imp_sampling(data=X_temp[sample_count:sample_count + 500, :],
            #                               data_number=500, score_func=self.score_net.functorch_score,
            #                               bandwidth=self.bandwidth,
            #                               grad_optim=optimizer,
            #                               # grad_optim=self.opt([X_filled], lr=self.lr),
            #                               iter_steps=self.sampling_step, mask_matrix=grad_mask[sample_count:sample_count + 500], clamp=clamp)
            X_filled = self.knew_imp_sampling(data=X_filled, data_number=n, score_func=self.score_net.functorch_score,
                                          bandwidth=self.bandwidth,
                                          grad_optim=optimizer,
                                          # grad_optim=self.opt([X_filled], lr=self.lr),
                                          iter_steps=self.sampling_step, mask_matrix=grad_mask, clamp=clamp)
            
            # X_filled = X_temp
            model_end_time = time.time()


            if X_true is not None:
                maes[i] = MAE(X_filled.detach().cpu().numpy(), X_true.detach().cpu().numpy(), mask.detach().cpu().numpy()).item()

                # if n <= OTLIM:
                #     M = mask.sum(1) > 0
                #     nimp = M.sum().item()
                #     dist = ((X_filled.detach().cpu().numpy()[M][:, None] - X_true.detach().cpu().numpy()[M]) ** 2).sum(2) / 2.
                #     wass = ot.emd2(np.ones(nimp) / nimp, np.ones(nimp) / nimp, dist)
                # else:
                wass = 0.0

            if verbose and ((i + 1) % report_interval == 0):

                if X_true is not None:
                    logging.info(f'Iteration {i + 1}:\t Loss: na\t '
                                 f'Validation MAE: {maes[i]:.4f}\t')
                    result_dict = {"hidden": str(self.mlp_hidden), "entropy_reg": self.entropy_reg,
                                   "bandwidth": self.bandwidth,
                                   "score_epoch": self.score_net_epoch,  "interval": i,
                                    "mae": maes[i], "wass": wass,
                                   "train_time": train_end_time - train_start_time,
                                   "imp_time": model_end_time - model_start_time}
                    result_list.append(result_dict)
                else:
                    logging.info(f'Iteration {i + 1}:\t Loss: na')

        # X_filled = X.detach().clone()
        # X_filled[mask.bool()] = imps

        if X_true is not None:
            return X_filled, result_list
        else:
            return X_filled
