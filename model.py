import torch
import torch.nn as nn
import torch.nn.functional as F
from prettytable import PrettyTable

#From Thong Nguyen on 
# https://stackoverflow.com/questions/49201236/
# check-the-total-number-of-parameters-in-a-pytorch-model
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

import torchvision.models as models
class Image_Res_Encoder(nn.Module):
    def __init__(self):
        super(Image_Res_Encoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[0:8])
        
    def forward(self, x):
        x = self.features(x)
        return x
    
#enc = Image_Res_Encoder()
#x = torch.randn(5,3,256,256)
#out = enc(x)
#print(out.shape)
#exit()

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class BlockEnc(nn.Module):
    def __init__(self,dim,n_heads,hidden_fc_dim=2048):
        super(BlockEnc, self).__init__()
        self.l_norm1 = nn.LayerNorm(dim)
        self.l_norm2 = nn.LayerNorm(dim)

        self.attention = nn.MultiheadAttention(dim,n_heads,batch_first=True,dropout=0.0)

        self.fc = nn.Sequential(
            nn.Linear(dim,int(hidden_fc_dim*2)),
            SwiGLU(),
            nn.Linear(hidden_fc_dim,dim)
            )

    def forward(self,x):
        attention_out,_ = self.attention(x,x,x)
        x = self.l_norm1(attention_out+x)
        
        fw = self.fc(x)
        x = self.l_norm2(fw+x)
        return x



class BlockDec(nn.Module):
    def __init__(self,dim,n_heads,hidden_fc_dim=2048):
        super(BlockDec, self).__init__()
        self.l_norm1 = nn.LayerNorm(dim)
        self.l_norm2 = nn.LayerNorm(dim)

        self.self_attention = nn.MultiheadAttention(dim,n_heads,batch_first=True,dropout=0.0)
        self.cross_attention = nn.MultiheadAttention(dim,n_heads,batch_first=True,dropout=0.0)

        self.fc = nn.Sequential(
            nn.Linear(dim,int(hidden_fc_dim*2)),
            SwiGLU(),
            nn.Linear(hidden_fc_dim,dim)
            )

    def forward(self,x):
        x, context = x
        attention_out,_ = self.self_attention(x,x,x)
        x = self.l_norm1(attention_out+x)

        cross_attention,_ = self.cross_attention(x,context,context)
        x = self.l_norm1(cross_attention+x)

        
        fw = self.fc(x)
        x = self.l_norm2(fw+x)
        return (x,context)


class ENCODER(nn.Module):
    def __init__(self,max_len,in_size=2048,dim=1024,heads=16,num_encoder=4):#dim=512 heads=16 num_decpder=16 for medium
        super(ENCODER, self).__init__()
        self.max_len = max_len

        self.embed = nn.Linear(in_size,dim)
        self.positional_embedding = nn.Embedding(self.max_len,dim)

        self.block = nn.Sequential(
            *[BlockEnc(dim,heads,hidden_fc_dim=dim*4)for _ in range(num_encoder)]
            )

        self.ln = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim,dim,bias=True)

    def forward(self,x):
        x = self.embed(x)
        bs,len,_ = x.shape

        positions = (torch.arange(0, len).unsqueeze(0).expand(bs, -1)).cuda()
        #print(self.positional_embedding(positions).shape, '0')
        #print(x.shape, '1')
        x = x + self.positional_embedding(positions)

        x = self.block(x)
        #print(x.shape,'2')

        x = self.ln(x)
        logits = self.fc(x)
        return logits

class DECODER(nn.Module):
    def __init__(self,space_size,dim,out_dim,num_decoder=4):
        super(DECODER, self).__init__()
        self.max_len = space_size[0]*space_size[1]*space_size[2]
        self.space_size = space_size
        self.positional_embedding = nn.Embedding(self.max_len,dim)

        
        self.block = nn.Sequential(
            *[BlockDec(dim,16,hidden_fc_dim=dim*4)for _ in range(num_decoder)]
            )

        self.fc = nn.Linear(dim,out_dim)

    def forward(self,kv):
        bs,_,_ = kv.shape
        positions = (torch.arange(0, self.max_len).unsqueeze(0).expand(bs, -1)).cuda()
        q = self.positional_embedding(positions)

        q,kv = self.block((q,kv))
        #print("WORKS")
        vox = self.fc(q)
        #print("WORKS1")
        return vox


def patchify(tensor, patch_height, patch_width):
    patches = tensor.unfold(2, patch_height, patch_height).unfold(3, patch_width, patch_width)
    patches = torch.flatten(patches,2,3).permute(0,2,1,3,4)
    patches = torch.flatten(patches,2)
    return patches

class Model(nn.Module):
    def __init__(self,shape=(17,17,5),que_len=8,num_rooms=512):
        super(Model, self).__init__()
        self.num_rooms = num_rooms
        self.shape = shape
        self.image_encoder = Image_Res_Encoder()
        self.encoder = ENCODER(int(64*que_len))
        self.decoder = DECODER(shape,1024,num_rooms)

        self.conv = nn.Sequential(
            nn.ConvTranspose3d(num_rooms,int(num_rooms/2),(1,1,4)),
            nn.SiLU(inplace=True),
            nn.ConvTranspose3d(int(num_rooms/2),int(num_rooms/4),(1,1,4)),
            nn.SiLU(inplace=True),
            nn.ConvTranspose3d(int(num_rooms/4),int(num_rooms/4),(1,1,4)),
            nn.SiLU(inplace=True),
            nn.ConvTranspose3d(int(num_rooms/4),1,(1,1,4)))
        

    def forward(self,img):
        #(batch,que_lengh,channels,img_size,img_size)
        _,que_lengh,_,_,_ = img.size()

        # Feed all frames through Image Encoder
        features=[]

        for i in range(que_lengh):
            x3 = self.image_encoder(img[:,i])
            x3 = patchify(x3,1,1)
            #print(x3.shape)
            #exit()
            #x5 = torch.flatten(x5,2).permute(0,2,1)
            #print(x5.shape)
            features.append(x3)

        x = torch.cat(features,1)
        print(x.shape)
        x = self.encoder(x)
        print(x.shape)
        x = self.decoder(x)
        x = x.permute(0,2,1)
        x = x.view((x.shape[0],x.shape[1],self.shape[0],self.shape[1],self.shape[2]))
        x = self.conv(x)
        return x

if __name__ == '__main__':
    x = torch.randn((1,8,3,256,256)).cuda()
    model= Model().cuda()
    count_parameters(model)
    print(x.shape)
    x = model(x)
    print(x.shape)
