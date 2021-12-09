import numpy as np
import torch
import os
import os.path as osp
from torch.autograd import Variable
from torchvision.utils import save_image
from .base_model import BaseModel
from . import net_utils
from .SCDis import SCDis
from .vgg import vgg16
from .losses import GANLoss, HistogramLoss
from .SCGen import SCGen

class SCGAN(BaseModel):
    def name(self):
        return 'SCGAN'
    def __init__(self,dataset):
        super(SCGAN, self).__init__()
        self.dataloader = dataset

    def initialize(self, opt):
        BaseModel.initialize(self, opt)    # 文件路径 要相对于 最开始执行的文件的路径
        self.lips = True
        self.eye = True
        self.skin = True
        self.num_epochs = opt.num_epochs
        self.num_epochs_decay = opt.epochs_decay
        self.g_lr = opt.g_lr
        self.d_lr = opt.d_lr
        self.g_step = opt.g_step
        self.beta1 = opt.beta1
        self.beta2 = opt.beta2
        self.img_size = opt.img_size
        self.lambda_idt = opt.lambda_idt
        self.lambda_A = opt.lambda_A
        self.lambda_B = opt.lambda_B
        self.lambda_his_lip = opt.lambda_his_lip
        self.lambda_his_skin_1 = opt.lambda_his_skin
        self.lambda_his_skin_2 = opt.lambda_his_skin
        self.lambda_his_eye = opt.lambda_his_eye
        self.lambda_vgg = opt.lambda_vgg
        self.snapshot_step = opt.snapshot_step
        self.save_step = opt.save_step
        self.log_step = opt.log_step
        self.result_path = opt.save_path
        self.snapshot_path = opt.snapshot_path
        self.d_conv_dim = opt.d_conv_dim
        self.d_repeat_num = opt.d_repeat_num
        self.norm1 = opt.norm1
        self.mask_A = {}
        self.mask_B = {}
        self.ispartial=opt.partial
        self.isinterpolation=opt.interpolation
        self.SCGen = SCGen(opt.ngf, opt.style_dim, opt.n_downsampling, opt.n_res, opt.mlp_dim, opt.n_componets, opt.input_nc, ispartial=opt.partial, isinterpolation=opt.interpolation)
        self.D_A = SCDis(self.img_size, self.d_conv_dim, self.d_repeat_num, self.norm1)
        self.D_B = SCDis(self.img_size, self.d_conv_dim, self.d_repeat_num, self.norm1)

        self.D_A.apply(net_utils.weights_init_xavier)
        self.D_B.apply(net_utils.weights_init_xavier)
        self.SCGen.apply(net_utils.weights_init_xavier)
        self.load_checkpoint()
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionL2 = torch.nn.MSELoss()

        self.criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor)
        self.vgg = vgg16(pretrained=False)
        self.criterionHis = HistogramLoss()

        self.g_optimizer = torch.optim.Adam(self.SCGen.parameters(), self.g_lr, [opt.beta1, opt.beta2])
        self.d_A_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_A.parameters()), opt.d_lr,
                                              [self.beta1, self.beta2])
        self.d_B_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_B.parameters()), opt.d_lr,
                                              [opt.beta1, opt.beta2])
        self.SCGen.cuda()
        self.vgg.cuda()
        self.criterionHis.cuda()
        self.criterionGAN.cuda()
        self.criterionL1.cuda()
        self.criterionL2.cuda()
        self.D_A.cuda()
        self.D_B.cuda()

        print('---------- Networks initialized -------------')
        # net_utils.print_network(self.SCGen)

    def load_checkpoint(self):
        G_path = os.path.join(self.snapshot_path ,'G.pth')
        if os.path.exists(G_path):
            dict=torch.load(G_path)
            self.SCGen.load_state_dict(dict)
            print('loaded trained generator {}..!'.format(G_path))
        D_A_path = os.path.join(self.snapshot_path, 'D_A.pth')
        if os.path.exists(D_A_path):
            self.D_A.load_state_dict(torch.load(D_A_path))
            print('loaded trained discriminator A {}..!'.format(D_A_path))

        D_B_path = os.path.join(self.snapshot_path, 'D_B.pth')
        if os.path.exists(D_B_path):
            self.D_B.load_state_dict(torch.load(D_B_path))
            print('loaded trained discriminator B {}..!'.format(D_B_path))


    def set_input(self, input):
        self.mask_A=input['mask_A']
        self.mask_B=input['mask_B']
        makeup=input['makeup_img']
        nonmakeup=input['nonmakeup_img']
        makeup_seg=input['makeup_seg']
        nonmakeup_seg=input['nonmakeup_seg']

        self.makeup=makeup
        self.nonmakeup=nonmakeup
        self.makeup_seg=makeup_seg
        self.nonmakeup_seg=nonmakeup_seg
        self.makeup_unchanged=input['makeup_unchanged']
        self.nonmakeup_unchanged=input['nonmakeup_unchanged']



    def to_var(self, x, requires_grad=False):
        if isinstance(x, list):
            return x
        if torch.cuda.is_available():
            x = x.cuda()
        if not requires_grad:
            return Variable(x, requires_grad=requires_grad)
        else:
            return Variable(x)

  #The training code is temporarily closed
    def train(self):
        pass

    def imgs_save(self, imgs_list):
        # saving results
        length = len(imgs_list)
        for i in range(0, length):
            imgs_list[i] = torch.cat(imgs_list[i], dim=3)   # torch.cat是将两个张量（tensor）拼接在一起
        imgs_list = torch.cat(imgs_list, dim=2)

        if not osp.exists(self.result_path):
            os.makedirs(self.result_path)

        save_path = os.path.join(self.result_path, '{}{}.jpg'.format("fpx" if self.ispartial else "fpx", "interpolation_" if self.isinterpolation else ""))
        save_image(self.de_norm(imgs_list.data), save_path, normalize=True)

    def log_terminal(self):
        log = " Epoch [{}/{}], Iter [{}/{}]".format(
            self.e+1, self.num_epochs, self.i+1, self.iters_per_epoch)

        for tag, value in self.loss.items():
            log += ", {}: {:.4f}".format(tag, value)
        print(log)


    def save_models(self):
        if not osp.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path)
        torch.save(
            self.SCGen.state_dict(),
            os.path.join(
                self.snapshot_path, '{}_{}_G.pth'.format(self.e + 1, self.i + 1)))
        torch.save(
            self.D_A.state_dict(),
            os.path.join(
                self.snapshot_path, '{}_{}_D_A.pth'.format(self.e + 1, self.i + 1)))
        torch.save(
            self.D_B.state_dict(),
            os.path.join(
                self.snapshot_path, '{}_{}_D_B.pth'.format(self.e + 1, self.i + 1)))


    def test(self):
            self.SCGen.eval()
            self.D_A.eval()
            self.D_B.eval()
            makeups = []
            makeups_seg = []
            nonmakeups=[]
            nonmakeups_seg = []
            for self.i, data in enumerate(self.dataloader):
                if (len(data) == 0):
                    print("No eyes!!")
                    continue
                # print(data)
                self.set_input(data)
                makeup, nonmakeup = self.to_var(self.makeup), self.to_var(self.nonmakeup),
                makeup_seg, nonmakeup_seg = self.to_var(self.makeup_seg), self.to_var(self.nonmakeup_seg)
                makeups.append(makeup)
                makeups_seg.append(makeup_seg)
                nonmakeups.append(nonmakeup)
                nonmakeups_seg.append(nonmakeup_seg)
            # print('data-ok')
            # source, ref1, ref2 = nonmakeups[0], makeups[0], makeups[1]
            source, ref1 = nonmakeups[0], makeups[0]
            # source_seg, ref1_seg, ref2_seg = nonmakeups_seg[0], makeups_seg[0], makeups_seg[1]
            source_seg, ref1_seg = nonmakeups_seg[0], makeups_seg[0]
            ref2 = ref1
            ref2_seg = ref1_seg
            with torch.no_grad():
                transfered = self.SCGen(source, source_seg, ref1, ref1_seg, ref2, ref2_seg)
                # print(type(transfered))
                # trans = torch.Tensor(transfered)   # 转换为 numpy 类型
            # print('torch-ok')
            if not self.ispartial and not self.isinterpolation:       #　当前　为这个
                results = [
                        [],
                        ]
                for i, img in zip(range(0, len(results)), transfered):
                    results[i].append(img)
                self.imgs_save(results)

            elif not self.ispartial and self.isinterpolation:
                results = [[source, ref1],
                           # [source, ref2],
                           [ref1, source],
                           # [ref2, source],
                           # [ref2, ref1]
                           ]
                for i, imgs in zip(range(0, len(results)-1), transfered):
                    for img in imgs:
                        results[i].append(img)
                for img in transfered[-1]:
                    results[-1].insert(1, img)
                results[-1].reverse()
                self.imgs_save(results)

            elif self.ispartial and not self.isinterpolation:
                results = [[source, ref1],
                           # [source, ref2],
                           # [source, ref1, ref2],
                           ]
                for i, imgs in zip(range(0, len(results)), transfered):
                    for img in imgs:
                        results[i].append(img)
                self.imgs_save(results)

            elif self.ispartial and self.isinterpolation:
                results = [[source, ref1],
                           [source, ref1],
                           [source, ref1],
                           # [source, ref2],
                           # [source, ref2],
                           # [source, ref2],
                           # [ref2, ref1],
                           # [ref2, ref1],
                           # [ref2, ref1],
                           ]
                for i, imgs in zip(range(0, len(results)-3), transfered):
                    for img in imgs:
                        results[i].append(img)
                for i, imgs in zip(range(len(results)-3, len(results)), transfered[-3:]):
                    for img in imgs:
                        results[i].insert(1, img)
                    results[i].reverse()
                self.imgs_save(results)



    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)