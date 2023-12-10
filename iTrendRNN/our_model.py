import torch
import torch.nn as nn
from config import Config as con
import numpy as np
import torch.nn.functional as F

##Here, since we employ a convolution with stride being 2, the height and width are 40.
##In your practice, you need to modify them according to the image size of your own data
height,width=40,40

class ConvGRUCell(nn.Module):
    def __init__(self, num_hidden, filter_size):
        super(ConvGRUCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1
        self.conv_x = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, padding=self.padding),
            nn.LayerNorm([num_hidden * 3, height, width])
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 2, kernel_size=filter_size, padding=self.padding),
            nn.LayerNorm([num_hidden * 2, height, width])
        )

        self.conv_zn = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, padding=self.padding),
            nn.LayerNorm([num_hidden, height, width])
        )

        self.conv_atth = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, padding=self.padding),
            nn.LayerNorm([num_hidden, height, width])
        )

        self.query = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.key = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.softmax = torch.nn.Softmax(dim=1)

    def att(self,n_t,h_t_list):
        q=self.query(n_t)#b,c,h,w
        l=len(h_t_list)
        s=[]
        r=[]

        for i in range(l):
            if(l>1 and i==0):continue
            h=h_t_list[i]
            r.append(h)
            k=self.key(h)#b,c,h,w
            s.append(q*k)
        s=torch.stack(s,dim=1)
        s=self.softmax(s)
        r = torch.stack(r, dim=1)
        res=torch.sum(s*r,dim=1)
        return res

    def forward(self, x_t, h_t_list):
        h_t=h_t_list[-1]
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)

        r_x, z_x, n_x = torch.split(x_concat, self.num_hidden, dim=1)
        r_h, n_h = torch.split(h_concat, self.num_hidden, dim=1)

        r_t = torch.sigmoid(r_x + r_h)
        n_t = torch.tanh(n_x + r_t * n_h)

        att_h=self.att(n_t,h_t_list)

        z_h=self.conv_atth(att_h)
        z_t = torch.sigmoid(z_x + z_h+self.conv_zn(n_t)+ self._forget_bias)

        h_new = (1-z_t)*n_t + z_t*att_h

        return h_new

class FlowRNNCell(nn.Module):
    def __init__(self,prior_distance):
        super(FlowRNNCell, self).__init__()
        self.num_hidden=64
        # height,width=40,40
        self.prior_distance=prior_distance
        self.query=nn.Conv2d(self.num_hidden,self.num_hidden,kernel_size=3,padding=1,bias=False)
        self.key = nn.Conv2d(self.num_hidden, self.num_hidden, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.gate=nn.Sequential(
            nn.Conv2d(self.num_hidden*3, self.num_hidden * 3, kernel_size=5, padding=2),
            nn.LayerNorm([self.num_hidden * 3, height, width])
        )

        self.ks=5
        if(prior_distance):
            self.dis = self.init_distance()

    def init_distance(self):
        dis = torch.zeros(self.ks, self.ks).cuda()
        certer_x = int((self.ks - 1) / 2)
        certer_y = int((self.ks - 1) / 2)
        for i in range(self.ks):
            for j in range(self.ks):
                ii = i - certer_x
                jj = j - certer_y
                tmp = (self.ks - 1) * (self.ks - 1)
                tmp = (ii * ii + jj * jj) / tmp + dis[i, j]
                dis[i, j] = torch.exp(-tmp)
        dis[certer_x, certer_y] = 0
        return dis

    def forward(self, x, f_t):
        b, c, h, w = x.shape
        ks = self.ks
        pad = ks // 2
        pad_x = F.pad(x, [pad, pad, pad, pad], mode='replicate')  # b,c,h+2p,w+2p
        ux = pad_x.unfold(2, ks, 1).unfold(3, ks, 1)  # b,c,h,w,ks,ks
        # res = x.unsqueeze(-1).unsqueeze(-1) - ux  # b,c,h,w,ks,ks
        res = ux-x.unsqueeze(-1).unsqueeze(-1)
        kr=res.permute(0,2,3,1,4,5)
        kr=kr.reshape(b*h*w,c,ks,ks)
        kr=self.key(kr)#
        kr=kr.reshape(b,h,w,c,ks*ks)
        q=self.query(f_t).permute(0,2,3,1)#b,h,w,c
        q=q.unsqueeze(3)#b,h,w,1,c
        s=torch.einsum('bhwij,bhwjk->bhwik', q, kr)#b,h,w,1,ks*ks

        if(self.prior_distance):
            ds=self.dis.view(1,self.ks*self.ks)
            s=s*ds

        s=self.softmax(s)
        s=s.reshape(b,h,w,ks,ks).unsqueeze(1)#b,1,h,w,ks,ks
        ff=res*s
        ff=torch.sum(ff,dim=(-1,-2))#b,c,h,w
        ff=torch.tanh(ff)

        conv_g=self.gate(torch.cat((x,f_t,ff),dim=1))
        g1,g2,g3=torch.split(conv_g, self.num_hidden, dim=1)
        gate=torch.sigmoid(g1+g2+g3)
        f_t_new=gate*f_t+(1-gate)*ff
        return f_t_new

class TrendRNNCell(nn.Module):
    def __init__(self, num_hidden, filter_size,prior_distance):
        super(TrendRNNCell, self).__init__()

        self.ft=FlowRNNCell(prior_distance)

        self.num_hidden = num_hidden
        self.padding = filter_size // 2

        ##sknet fusion
        self.ds = nn.Conv2d(num_hidden, num_hidden // 2, kernel_size=3, padding=1, stride=2, bias=False)
        self.fcs = nn.ModuleList([])
        for i in range(3):
            self.fcs.append(
                nn.Sequential(
                    nn.ConvTranspose2d(num_hidden // 2, num_hidden, kernel_size=3, padding=1, stride=2, bias=False,output_padding=1),
                    nn.Tanh())
            )
        self.softmax = nn.Softmax(dim=1)

    def trend_fusion(self,fo_t,ho_t,ft):
        # f_trend=self.fusion(torch.cat((fo_t,ho_t,ft),dim=1))

        cat = torch.stack([fo_t, ho_t, ft], dim=1)  # b,3,c,h,w
        avg = torch.mean(cat, dim=1)  # b,c,h,w
        mp = []
        avg = self.ds(avg)
        for i in range(3):
            tmp = self.fcs[i](avg)
            mp.append(tmp)
        mp = torch.stack(mp, dim=1)  # b,3,c,h,w
        mp = self.softmax(mp)  # b,3,c,h,w
        res = (cat * mp).sum(dim=1)

        return res

    def forward(self, xh,h_t, fo_t,ho_t,f_t):
        #x_t,h_t:b,c,h,w
        f_t_new=self.ft(xh,f_t)
        trend=self.trend_fusion(fo_t,ho_t,f_t_new)
        h_new=h_t+trend
        return h_new,f_t_new

class TrendRNN(nn.Module):
    def __init__(self,args):
        super(TrendRNN, self).__init__()
        self.hidden_num=64
        self.MSE_criterion = nn.MSELoss(size_average=True)
        self.args=args

        self.ec=nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.Conv2d(32, self.hidden_num, kernel_size=3, padding=1, stride=2, bias=False)
        )
        self.trendrnn=TrendRNNCell(self.hidden_num,5,self.args.prior_distance)
        self.dc=nn.Sequential(
            nn.ConvTranspose2d(self.hidden_num, 32, kernel_size=3, padding=1, stride=2, bias=False,output_padding=1),
            nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1, stride=1, bias=False)
        )

        self.conv_xh = nn.Conv2d(self.hidden_num * 2, self.hidden_num, kernel_size=3, padding=1)

        self.convgru=ConvGRUCell(self.hidden_num,5)
        self.conv_ho=nn.Conv2d(self.hidden_num, self.hidden_num, kernel_size=3, padding=1)

        if(self.args.high_order_trend):
            self.convgru2 = ConvGRUCell(self.hidden_num, 5)
            self.conv_ho2 = nn.Conv2d(self.hidden_num, self.hidden_num, kernel_size=3, padding=1)

        self.softmax = torch.nn.Softmax(dim=1)

    def ho_fusion(self,ho_bank):
        #mean fusion
        ho_bank=torch.stack(ho_bank,dim=0)
        ho_bank=torch.mean(ho_bank,dim=0)
        return ho_bank

    def forward(self,frames,il=6,input_flag=None,train_flag=True):
        #inpout_flag:b,ol-1,h,w,1
        b,l,h,w,c=frames.shape
        res=[]

        h_t = torch.zeros((b, self.hidden_num, height, width)).cuda()
        f_t = torch.zeros((b, self.hidden_num, height, width)).cuda()

        hho_t1 = torch.zeros((b, self.hidden_num, height, width)).cuda()
        hho_t1_list= [hho_t1]
        ho_t1_list= []

        if(self.args.high_order_trend):
            hho_t5 = torch.zeros((b, self.hidden_num, height, width)).cuda()
            hho_t5_list = [hho_t5]
            ho_t5_list = []

        xh_list = []
        for t in range(con.input_length+con.output_length-1):
            if(t<il):xt=frames[:,t].permute(0,3,1,2)#b,1,h,w
            elif(train_flag):
                xt= input_flag[:, t - con.input_length] * frames[:, t] + (1 - input_flag[:, t - con.input_length]) * (res[-1].permute(0,2,3,1))
                xt=xt.permute(0,3,1,2)
            else:xt=res[-1]
            xt = self.ec(xt)

            if(t<=1):xh=torch.cat((xt,xt),dim=1)
            else:xh=torch.cat((xt,h_t),dim=1)
            xh=self.conv_xh(xh)
            xh_list.append(xh)

            if (t == 0): continue

            ### we have skip t=0

            ##FO_trend
            fo_t = xh_list[-1] - xh_list[-2]

            ##HO_trend
            ho_bank=[]
            hho_t1 = self.convgru(fo_t, hho_t1_list)
            hho_t1_list.append(hho_t1)
            ho_t1 = self.conv_ho(hho_t1)
            ho_bank.append(ho_t1)
            ho_t1_list.append(ho_t1)

            if (t >= self.args.input_length-1 and self.args.high_order_trend):
                hho_t5 = self.convgru2(fo_t, hho_t5_list)
                hho_t5_list.append(hho_t5)
                ho_t5 = self.conv_ho2(hho_t5)
                ho_bank.append(ho_t5)
                ho_t5_list.append(ho_t5)

            ho_t=self.ho_fusion(ho_bank)
            h_t,f_t = self.trendrnn(xh, h_t, fo_t, ho_t,f_t)
            nxt = self.dc(h_t)
            res.append(nxt)

        ##compute trend true
        trend_true=[]
        for i in range(1, len(xh_list)): trend_true.append(xh_list[i] - xh_list[i - 1])
        if(train_flag==False):xt=res[-1]
        else:xt=frames[:,-1].permute(0,3,1,2)
        xt = self.ec(xt)
        xh=torch.cat((xt,h_t),dim=1)
        xh=self.conv_xh(xh)
        trend_true.append(xh-xh_list[-1])
        trend_true=torch.stack(trend_true,dim=1).permute(0,1,3,4,2)[:,1:]

        ##calculate loss
        res = torch.stack(res, dim=1).permute(0, 1, 3, 4, 2)  # b,10,h,w,1
        loss = self.MSE_criterion(res[:, -con.output_length - 2:], frames[:, -con.output_length - 2:])
        loss=loss+self.MSE_criterion(torch.stack(ho_t1_list,dim=1).permute(0,1,3,4,2), trend_true)
        if(self.args.high_order_trend):
            loss = loss + self.MSE_criterion(torch.stack(ho_t5_list,dim=1).permute(0,1,3,4,2), trend_true[:, con.input_length-2:])
        return res,loss
