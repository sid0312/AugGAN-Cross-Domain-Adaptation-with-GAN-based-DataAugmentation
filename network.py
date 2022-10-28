import cv2
import itertools
import os
import tqdm
from models import *
from lossnhistory import *
# from utils.logger import Logger
import utils.util as util
from utils.image_pool import ImagePool

class network():
    def __init__(self, params):
        super(network, self).__init__()
        self.Tensor = torch.cuda.FloatTensor
        self.configurate(params['net'])

        self.fake_pool_x = ImagePool(self.pool_size)
        self.fake_pool_y = ImagePool(self.pool_size)
        self.input_x = self.Tensor(self.batch_size, 3, 256, 256)
        self.input_y = self.Tensor(self.batch_size, 3, 256, 256)
        self.target_x = self.Tensor(self.batch_size, 1, 256, 256)
        self.target_y = self.Tensor(self.batch_size, 1, 256, 256)

        # self.tf_summary = Logger('./logs', self.name)

        self.enc_x = Encoder(**params['enc_x']).cuda()
        self.enc_y = Encoder(**params['enc_y']).cuda()
        self.mul_gen_x = Multitask_Generator(**params['gen_x']).cuda()
        self.mul_gen_y = Multitask_Generator(**params['gen_y']).cuda()

        self.dis_x = NLayerDiscriminator(**params['dis']).cuda()
        self.dis_y = NLayerDiscriminator(**params['dis']).cuda()

        self.criterionGAN = GANLoss()
        self.criterionCyC = torch.nn.L1Loss()
        # self.criterionSeg = Segmentation_Loss()

        self.optimizer_G = torch.optim.Adam(itertools.chain(self.enc_x.parameters(), self.mul_gen_x.parameters(), 
                                                            self.enc_y.parameters(), self.mul_gen_y.parameters()),
                                                            lr=self.lr, betas=(0.5,0.999))
        self.optimizer_D_A = torch.optim.Adam(self.dis_x.parameters(), lr=self.lr, betas=(0.5,0.999))
        self.optimizer_D_B = torch.optim.Adam(self.dis_y.parameters(), lr=self.lr, betas=(0.5,0.999))
    
    def set_input(self, data):
        AtoB = self.which_direction == 'AtoB'
        input_x = data.get('A') if AtoB else data.get('B')
        input_y = data.get('B') if AtoB else data.get('A')
        label_x = data.get('A_label') if AtoB else data.get('B_label')
        label_y = data.get('B_label') if AtoB else data.get('A_label')

        self.input_x.resize_(input_x.size()).copy_(input_x)
        self.input_y.resize_(input_y.size()).copy_(input_y)

        self.real_x = Variable(self.input_x)
        self.real_y = Variable(self.input_y)
        self.target_x = Variable(label_x).cuda()
        self.target_y = Variable(label_y).cuda()
        self.image_paths = data.get('A_Paths') if AtoB else data.get('B_Paths')
    
    def Encode(self):
        self.z_x = self.enc_x(self.real_x)
        self.z_y = self.enc_y(self.real_y)    

    def Generate(self):
        self.fake_y = self.mul_gen_x(self.z_x)
        self.fake_x = self.mul_gen_y(self.z_y)
    
    def optimize_params(self):
        #Optimize Encoder_Parsing Pair
#        self.optimizer_E.zero_grad()
#        self.backward_encoder()
#        self.optimizer_E.step()
        self.Encode()
        self.Generate()
        #Optimize Generator
        self.optimizer_G.zero_grad()
        self.backward_generator()
        self.optimizer_G.step()
        #Optimizer Discriminator A
        self.optimizer_D_A.zero_grad()
        self.backward_Discriminator_A()
        self.optimizer_D_A.step()
		#Optimizer Discriminator B
        self.optimizer_D_B.zero_grad()
        self.backward_Discriminator_B()
        self.optimizer_D_B.step()

    def compute_loss(self):
        self.Encode()
        self.Generate()
        #Optimize Generator
        self.optimizer_G.zero_grad()
        self.backward_generator(backprop=False)

        self.optimizer_D_A.zero_grad()
        self.backward_Discriminator_A(backprop=False)

        self.optimizer_D_B.zero_grad()
        self.backward_Discriminator_B(backprop=False)

    def backward_generator(self, backprop=True):        
        pred_fake = self.dis_x.forward(self.fake_y)
        self.loss_G_x = self.criterionGAN(pred_fake, True) 
        # self.loss_seg_x, _ = self.criterionSeg(self.parse_x, self.target_x)
        # self.loss_seg_x *= self.lambda_seg_x

        pred_fake = self.dis_y.forward(self.fake_x)
        self.loss_G_y = self.criterionGAN(pred_fake, True)
        # self.loss_seg_y, _ = self.criterionSeg(self.parse_y, self.target_y)
        # self.loss_seg_y *= self.lambda_seg_y

        self.rec_x = self.mul_gen_y(self.enc_y(self.fake_y))
        self.loss_cycle_x = self.criterionCyC(self.rec_x, self.real_x) * self.lambda_A   
        
        self.rec_y = self.mul_gen_x(self.enc_x(self.fake_x))
        self.loss_cycle_y = self.criterionCyC(self.rec_y, self.real_y) * self.lambda_B

        #L2 Regularization
        l2_reg_x = 0
        for param_s, param_p in zip(self.mul_gen_x.shared_x.parameters(), self.mul_gen_x.shared_y.parameters()):
            diff = param_s - param_p
            l2_reg_x += (diff.norm(2))*self.lambda_ws_shared
        for param_s, param_p in zip(self.mul_gen_x.decoder_x.c_layers.parameters(), self.mul_gen_x.decoder_y.c_layers.parameters()):
            diff = param_s - param_p
            l2_reg_x += (diff.norm(2))*self.lambda_ws_decoder
        # self.loss_ws_x = l2_reg_x
        l2_reg_y = 0
        for param_s, param_p in zip(self.mul_gen_y.shared_x.parameters(), self.mul_gen_y.shared_y.parameters()):
            diff = param_s - param_p
            l2_reg_y += (diff.norm(2))*self.lambda_ws_shared
        for param_s, param_p in zip(self.mul_gen_y.decoder_x.c_layers.parameters(), self.mul_gen_y.decoder_y.c_layers.parameters()):
            diff = param_s - param_p
            l2_reg_x += (diff.norm(2))*self.lambda_ws_decoder
        # self.loss_ws_y = l2_reg_y

        # loss_G = self.loss_G_x + self.loss_G_y + self.loss_cycle_x + self.loss_cycle_y + self.loss_seg_x + self.loss_seg_y + self.loss_ws_x + self.loss_ws_y

        loss_G = self.loss_G_x + self.loss_G_y + self.loss_cycle_x + self.loss_cycle_y
        if backprop:
            loss_G.backward()

    def backward_D_basic(self, netD, real, fake, backprop=True):
    	#Real
        pred_real = netD.forward(real)
        loss_D_real = self.criterionGAN(pred_real, True)
		#Fake
        pred_fake = netD.forward(fake)
        loss_D_fake = self.criterionGAN(pred_fake, False)

        loss_D = (loss_D_real + loss_D_fake)/2
        if backprop:
            loss_D.backward()
        return loss_D

    def backward_Discriminator_A(self, backprop=True):
        fake_y = self.fake_pool_y.query(self.fake_y)
        self.loss_D_x = self.backward_D_basic(self.dis_x, self.real_y, fake_y, backprop)
 
    def backward_Discriminator_B(self, backprop=True):
        fake_x = self.fake_pool_x.query(self.fake_x)
        self.loss_D_y = self.backward_D_basic(self.dis_y, self.real_x, fake_x, backprop)  

    def save(self, epoch):
        self.folder_path = "../output"
        util.mkdirs(self.folder_path)
        self.save_network(self.enc_x, f'enc_x_{epoch}', self.folder_path)
        self.save_network(self.enc_y, f'enc_y_{epoch}', self.folder_path)
        self.save_network(self.mul_gen_x, f'gen_x_{epoch}', self.folder_path)
        self.save_network(self.mul_gen_y, f'gen_y_{epoch}', self.folder_path)
        self.save_network(self.dis_x, f'dis_x_{epoch}', self.folder_path)   
        self.save_network(self.dis_y, f'dis_y_{epoch}', self.folder_path)

    def save_model(self, epoch):
        self.folder_path = "../output"
        util.mkdirs(self.folder_path)
        self.save_script(torch.jit.script(self.enc_x), f'enc_x_{epoch}', self.folder_path)
        self.save_script(torch.jit.script(self.enc_y), f'enc_y_{epoch}', self.folder_path)
        self.save_script(torch.jit.script(self.mul_gen_x), f'gen_x_{epoch}', self.folder_path)
        self.save_script(torch.jit.script(self.mul_gen_y), f'gen_y_{epoch}', self.folder_path)
        self.save_script(torch.jit.script(self.dis_x), f'dis_x_{epoch}', self.folder_path)
        self.save_script(torch.jit.script(self.dis_y), f'dis_y_{epoch}', self.folder_path)

    def summary_image(self, tag, output, epoch):
        def to_numpy(imgs):
            return imgs.squeeze().cpu().numpy()
        imgs = []
        for i, img in enumerate(output):
            imgs.append(to_numpy(img))
        # self.tf_summary.image(tag, imgs, epoch)
	
    def sumup_image(self, epoch):
        self.summary_image('real_x', self.real_x.data, epoch)
        self.summary_image('real_y', self.real_y.data, epoch)
        self.summary_image('fake_y', self.fake_y.data, epoch)
        self.summary_image('fake_x', self.fake_x.data, epoch)
        _, parse_x = torch.max(self.parse_x, 1)
        self.summary_image('parse_x', parse_x.data, epoch)
        _, parse_y = torch.max(self.parse_y, 1)
        self.summary_image('parse_y', parse_y.data, epoch)
        self.summary_image('rec_A', self.rec_x.data, epoch)
        self.summary_image('rec_B', self.rec_y.data, epoch)
    
    def current_loss(self):
        # return {'loss_g_x': self.loss_G_x.item(), 'loss_g_y': self.loss_G_y.item(),
        #         'loss_cyc_x': self.loss_cycle_x.item(), 'loss_cyc_y': self.loss_cycle_y.item(),
        #         'loss_seg_x': self.loss_seg_x.item(), 'loss_seg_y': self.loss_seg_y.item(),
        #         'loss_d_x': self.loss_D_x.item(), 'loss_d_y': self.loss_D_y.item(),
        #         'loss_ws_x': self.loss_ws_x, 'loss_ws_y': self.loss_ws_y}

        return {'loss_g_x': self.loss_G_x.item(), 'loss_g_y': self.loss_G_y.item(),
                'loss_cyc_x': self.loss_cycle_x.item(), 'loss_cyc_y': self.loss_cycle_y.item(),
                'loss_d_x': self.loss_D_x.item(), 'loss_d_y': self.loss_D_y.item()}

    def save_network(self, network, network_label, folder_path):
        save_filename = '%s_net.pth' % network_label
        save_path = os.path.join(folder_path, save_filename)
        torch.save(network.state_dict(), save_path)
    
    def save_script(self, network, network_label, folder_path):
        save_filename = '%s_net.pt' % network_label
        save_path = os.path.join(folder_path, save_filename)
        network.save(save_path)

    def load_from_checkpoint(self, checkpoint_path, epoch):
        self.enc_x.load_state_dict(torch.load(os.path.join(checkpoint_path, f'enc_x_{epoch}_net.pth')))
        self.enc_y.load_state_dict(torch.load(os.path.join(checkpoint_path, f'enc_y_{epoch}_net.pth')))
        self.mul_gen_x.load_state_dict(torch.load(os.path.join(checkpoint_path, f'gen_x_{epoch}_net.pth')))
        self.mul_gen_y.load_state_dict(torch.load(os.path.join(checkpoint_path, f'gen_y_{epoch}_net.pth')))
        self.dis_x.load_state_dict(torch.load(os.path.join(checkpoint_path, f'dis_x_{epoch}_net.pth')))
        self.dis_y.load_state_dict(torch.load(os.path.join(checkpoint_path, f'dis_y_{epoch}_net.pth')))

    def load_model_from_checkpoint(self, checkpoint_path, epoch):
        self.enc_x = torch.jit.load(os.path.join(checkpoint_path, f'enc_x_{epoch}_net.pt'))
        self.enc_y = torch.jit.load(os.path.join(checkpoint_path, f'enc_y_{epoch}_net.pt'))
        self.mul_gen_x = torch.jit.load(os.path.join(checkpoint_path, f'gen_x_{epoch}_net.pt'))
        self.mul_gen_y = torch.jit.load(os.path.join(checkpoint_path, f'gen_y_{epoch}_net.pt'))
        self.dis_x = torch.jit.load(os.path.join(checkpoint_path, f'dis_x_{epoch}_net.pt'))
        self.dis_y = torch.jit.load(os.path.join(checkpoint_path, f'dis_y_{epoch}_net.pt'))

    def configurate(self, params):
        for k,v in params.items():
            cmd = "self." + k + "=" + repr(v)
            exec(cmd)

    def forward_pass(self):
        self.compute_loss()
        return self.real_x.detach(), self.rec_x.detach(), self.real_y.detach(), self.rec_y.detach()

    def generate_output(self):
        y_fake = self.mul_gen_x(self.enc_x(self.real_x))
        x_fake = self.mul_gen_y(self.enc_y(self.real_y))
        x_rec = self.mul_gen_y(self.enc_y(y_fake))
        y_rec = self.mul_gen_x(self.enc_x(x_fake))
        return x_rec, y_rec, x_fake, y_fake, self.real_x, self.real_y