import torch
from barbar import Bar
import os
import models


class EarlyStopper:
    def __init__(self, patience=1):
        self.patience = patience
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class TrainerDeepSVDD:
    def __init__(self, in_channels, num_filters, kernel_size, bottleneck_dim, device):
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.bottleneck_dim = bottleneck_dim
        self.device = device
        self.weights_folder = 'svdd_weights'
        os.makedirs(self.weights_folder, exist_ok=True)

    def weights_init_normal(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1 and classname != 'Conv':
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("Linear") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    
    def pretrain(self, dataloader, val_loader, num_epochs):
        """Pretraining the weights for the deep SVDD network using autoencoder"""

        pretrained_path = os.path.join(self.weights_folder, 'pretrained_autoencoder.pth')
        if os.path.exists(pretrained_path):
            print("Loading pretrained autoencoder weights...")
            ae = torch.load(pretrained_path).to(self.device)
        else:
            early_stopper = EarlyStopper(patience=3)
            ae = models.Autoencoder(self.in_channels, self.num_filters, self.kernel_size, self.bottleneck_dim).to(self.device)
            ae.apply(self.weights_init_normal)
            optimizer = torch.optim.Adam(ae.parameters(), 1e-4, weight_decay=0.5e-3)
            ae.train()
            for epoch in range(num_epochs):
                total_loss = 0
                total_val_loss = 0
                for x in Bar(dataloader):
                    x = x.float().to(self.device)
                    optimizer.zero_grad()
                    x_hat = ae(x)
                    reconst_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
                    reconst_loss.backward()
                    optimizer.step()
                    total_loss += reconst_loss.item()

                with torch.no_grad():
                    for x in val_loader:
                        x = x.float().to(self.device)
                        x_hat = ae(x)
                        val_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
                        total_val_loss += float(val_loss.detach().to("cpu"))

                if early_stopper.early_stop(total_val_loss):
                    break
                print('Pretraining Autoencoder... Epoch: {}, Loss: {:.3f} Val_Loss: {:.3f}'.format(
                       epoch, total_loss/len(dataloader), total_val_loss/len(val_loader)))

            torch.save(ae, pretrained_path)
            self.save_weights_for_DeepSVDD(ae, dataloader)
        self.ae = ae
    
    def save_weights_for_DeepSVDD(self, model, dataloader):
        """Initialize Deep SVDD weights using the encoder weights of the pretrained autoencoder."""
        net = models.DeepSVDDEncoder(self.in_channels, self.num_filters, self.kernel_size, self.bottleneck_dim).to(self.device)
        c = self.set_c(model, net, dataloader)
        state_dict = model.state_dict()
        net.load_state_dict(state_dict, strict=False)
        torch.save({'center': c.cpu().data.numpy().tolist(),
                    'net_dict': net.state_dict()}, os.path.join(self.weights_folder, 'pretrained_parameters.pth'))
    
    def set_c(self, model, net,  dataloader, eps=0.1):
        """Initializing the center for the hypersphere"""
        model.eval()
        z_ = []
        with torch.no_grad():
            for x in dataloader:
                x = x.float().to(self.device)
                z = net(x)
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c

    def train(self, dataloader, num_epochs):
        """Training the Deep SVDD model"""

        train_path = os.path.join(self.weights_folder, 'trained_svdd_model.pth')
        if os.path.exists(train_path):
            print("Loading trained SVDD model weights...")
            state_dict = torch.load(train_path)
            self.net = models.DeepSVDDEncoder(self.in_channels, self.num_filters, self.kernel_size, self.bottleneck_dim).to(self.device)
            self.net.load_state_dict(state_dict['net_dict'])
            self.c = torch.Tensor(state_dict['center']).to(self.device)
        else:
            net = models.DeepSVDDEncoder(self.in_channels, self.num_filters, self.kernel_size, self.bottleneck_dim).to(self.device)
            pretrained_path = os.path.join(self.weights_folder, 'pretrained_parameters.pth')
            state_dict = torch.load(pretrained_path)
            net.load_state_dict(state_dict['net_dict'])
            c = torch.Tensor(state_dict['center']).to(self.device)

            optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0.5e-6)
            net.train()
            for epoch in range(num_epochs):
                total_loss = 0
                for x in Bar(dataloader):
                    x = x.float().to(self.device)
                    optimizer.zero_grad()
                    z = net(x)
                    loss = torch.mean(torch.sum((z - c) ** 2, dim=1))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print('Training Deep SVDD... Epoch: {}, Loss: {:.3f}'.format(epoch, total_loss/len(dataloader)))
            self.net = net
            self.c = c
            torch.save({'center': c.cpu().data.numpy().tolist(),
                        'net_dict': net.state_dict()}, train_path)

    def eval(self, dataloader):
        scores = []
        self.net.eval()
        print('Testing...')
        with torch.no_grad():
            for x in dataloader:
                x = x.float().to(self.device)
                z = self.net(x)
                score = torch.sum((z - self.c) ** 2, dim=1)
                scores.append(score.detach().cpu())
        scores = torch.cat(scores).numpy()
        return scores.flatten()
