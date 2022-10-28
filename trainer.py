from network import *
# from scores import *
import matplotlib.pyplot as plt
import utils.util as util
from torchmetrics.image.fid import FrechetInceptionDistance


class trainer():
    def __init__(self, network, save_path, name, description):
        super(trainer, self).__init__()
        self.network = network
        self.save_path = save_path
        self.name = name
        self.description = description
        self.writefile = '../logs.txt'
        self.val_writefile = '../val_logs.txt'
        self.fid_writefile = '../fid_logs.txt'

    def train(self, train_data_loader, val_data_loader, epochs, test_data_loader=None):
        for epoch in range(1, epochs+1):
            self.network.enc_x.train()
            self.network.enc_y.train()
            self.network.mul_gen_x.train()
            self.network.mul_gen_y.train()
            self.network.dis_x.train()
            self.network.dis_y.train()
            print('currently running.....: {}'.format(self.name))
            print('description: {}'.format(self.description))
            self.epoch = epoch
            self.network.folder_path = self.save_path.format(self.network.name, epoch)

            progress = tqdm.tqdm(train_data_loader)
            hist = EpochHistory(length=len(progress))
            for data in progress:
                self.network.set_input(data)
                self.network.optimize_params()
                loss = self.network.current_loss()
                hist.add(loss)
                progress.set_description('Epoch #%d' % self.epoch)
                progress.set_postfix(
                    g_x='%.04f' % loss.get('loss_g_x'), g_y='%.04f' % loss.get('loss_g_y'),
                    seg_x='%.04f' % loss.get('loss_seg_x'), seg_y='%.04f' % loss.get('loss_seg_y'),
                    d_x='%.04f' % loss.get('loss_d_x'), d_y='%.04f' % loss.get('loss_d_y'),
                    ws_x='%.04f' % loss.get('loss_ws_x'), ws_y='%.04f' % loss.get('loss_ws_y'))
                            
            metrics = hist.metric()
            epoch_stats = '---> Epoch# %d summary loss g_x:{loss_g_x:.4f}, g_y:{loss_g_y:.4f}, seg_x:{loss_seg_x:.4f}, seg_y:{loss_seg_y:.4f}, d_x:{loss_d_x:.4f}, d_y:{loss_d_y:.4f}, ws_x:{loss_ws_x:.4f}, ws_y:{loss_ws_y:.4f}'.format(self.epoch, **metrics)
                  
            with open(self.writefile,'a') as f:
                f.write(epoch_stats+"\n")
            
            # self.network.sumup_image(self.epoch)
            # for k,v in metrics.items():
            #     self.network.tf_summary.scalar(k,v, self.epoch)
            # if self.epoch % 10 == 0:
            # self.network.save(self.epoch)s
            # self.network.save(0)
            self.network.save_model("latest")
            # if epoch > self.network.niter:
            #     self.network.update_learning_rate()

            self.validate(val_data_loader)
            if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
                self.visualize(test_data_loader, epoch)

    def validate(self, data_loader):
        self.network.enc_x.eval()
        self.network.enc_y.eval()
        self.network.mul_gen_x.eval()
        self.network.mul_gen_y.eval()
        self.network.dis_x.eval()
        self.network.dis_y.eval()
        with torch.no_grad():
            progress = tqdm.tqdm(data_loader)
            hist = EpochHistory(length=len(progress))
            for data in progress:
                self.network.set_input(data)
                self.network.compute_loss()
                loss = self.network.current_loss()
                hist.add(loss)
                progress.set_description('Epoch #%d' % self.epoch)
                progress.set_postfix(
                    g_x='%.04f' % loss.get('loss_g_x'), g_y='%.04f' % loss.get('loss_g_y'),
                    seg_x='%.04f' % loss.get('loss_seg_x'), seg_y='%.04f' % loss.get('loss_seg_y'),
                    d_x='%.04f' % loss.get('loss_d_x'), d_y='%.04f' % loss.get('loss_d_y'),
                    ws_x='%.04f' % loss.get('loss_ws_x'), ws_y='%.04f' % loss.get('loss_ws_y'))
            metrics = hist.metric()
            epoch_stats = '---> Epoch# {} summary loss g_x:{loss_g_x:.4f}, g_y:{loss_g_y:.4f}, d_x:{loss_d_x:.4f}, d_y:{loss_d_y:.4f}, ws_x:{loss_ws_x:.4f}, ws_y:{loss_ws_y:.4f}'.format(self.epoch, **metrics)      
            with open(self.val_writefile,'a') as f:
                f.write(epoch_stats+"\n")
        self.network.enc_x.train()
        self.network.enc_y.train()
        self.network.mul_gen_x.train()
        self.network.mul_gen_y.train()
        self.network.dis_x.train()
        self.network.dis_y.train()

    def evaluate(self, data_loader):
        self.network.enc_x.eval()
        self.network.enc_y.eval()
        self.network.mul_gen_x.eval()
        self.network.mul_gen_y.eval()
        self.network.dis_x.eval()
        self.network.dis_y.eval()
        X, Rec_X, Y, Rec_Y = [], [], [], []
        fid = FrechetInceptionDistance(feature=64, compute_on_cpu=True)
        
        with torch.no_grad():
            progress = tqdm.tqdm(data_loader)
            for idx, data in enumerate(progress):
                self.network.set_input(data)
                x, rec_x, y, rec_y = self.network.forward_pass()
                X.append(x)
                Rec_X.append(rec_x)
                Y.append(y)
                Rec_Y.append(rec_y)
            
            fid.update(torch.cat(X, 0).to(dtype=torch.uint8, device=torch.device('cpu')), real=True)
            fid.update(torch.cat(Rec_X, 0).to(dtype=torch.uint8, device=torch.device('cpu')), real=False)
            x_fid = fid.compute()
            fid.reset()
            
            
            fid.update(torch.cat(Y, 0).to(dtype=torch.uint8, device=torch.device('cpu')), real=True)
            fid.update(torch.cat(Rec_Y, 0).to(dtype=torch.uint8, device=torch.device('cpu')), real=False)
            y_fid = fid.compute()
            fid.reset()
            
            with open(self.fid_writefile,'a') as f:
                f.write("Mean FIDs: X - {} Y - {}".format(x_fid, y_fid)+"\n")
        
        self.network.enc_x.train()
        self.network.enc_y.train()
        self.network.mul_gen_x.train()
        self.network.mul_gen_y.train()
        self.network.dis_x.train()
        self.network.dis_y.train()

    def visualize(self, data_loader, epoch=1):
        self.network.enc_x.eval()
        self.network.enc_y.eval()
        self.network.mul_gen_x.eval()
        self.network.mul_gen_y.eval()
        self.network.dis_x.eval()
        self.network.dis_y.eval()
        util.mkdirs("./plots")
        with torch.no_grad():
            data =  next(iter(data_loader))
            self.network.set_input(data)
            x_rec, y_rec, x_fake, y_fake, x, y = self.network.generate_output()
            self.plot(x_rec,f'./plots/x_rec_{epoch}.png')
            self.plot(y_rec,f'./plots/y_rec_{epoch}.png')
            self.plot(x_fake,f'./plots/x_fake_{epoch}.png')
            self.plot(y_fake,f'./plots/y_fake_{epoch}.png')
            self.plot(x,f'./plots/x_{epoch}.png')
            self.plot(y,f'./plots/y_{epoch}.png')

        self.network.enc_x.train()
        self.network.enc_y.train()
        self.network.mul_gen_x.train()
        self.network.mul_gen_y.train()
        self.network.dis_x.train()
        self.network.dis_y.train()
            
    def plot(self, x, name):
        imgs = x.detach().to('cpu').permute(0,2,3,1).numpy()
        img = imgs[0]
        np.save(name,imgs)
        img = (img - np.min(img))
        img = img/np.max(img)
        plt.imshow(img)
        plt.savefig(name,dpi=300)