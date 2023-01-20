import os
from icecream import ic
import json
from torch.utils.data import Dataset, DataLoader
import PIL.Image as Image
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import random_split
import random
import torch.nn.functional as F
import wandb
import numpy as np
from copy import deepcopy

class Args:
    def __init__(self):
        self.path = 'medium'
        self.hidden_size = 512
        self.epoch = 1000
        self.batch_size = 32
        self.extra_crop = True
        self.crop = False
        self.normalize = 'batch'
        self.gpu_id = 0
        self.cuda = True
        self.checkpoint_path = 'wandb/run-20220625_053407-1z3rx3fk/files/epoch10.pth'
        self.resnet_path = "wandb/run-20220624_135259-1vp1y0u3/files/resnet_epoch6.pth"
        self.use_attn = True
        self.lr = 1e-4
        self.train_resnet = False
        self.use_gru = True
        self.fix_resnet = False
        self.train_both = True
        self.train_all = True
        self.inference = 'new'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        self.config = {
            'hidden_size':self.hidden_size,
            'epoch':self.epoch,
            'batch_size':self.batch_size,
            'crop':self.crop,
            'normalize':self.normalize,
            'gpu_id':self.gpu_id,
            'cuda':self.cuda,
            'checkpoint':self.checkpoint_path,
            'lr':self.lr,
            'use_gru':self.use_gru,
            'use_attn':self.use_attn,
            'extra_crop':self.extra_crop,
            'train_resnet':self.train_resnet,
            'fix_resnet':self.fix_resnet,
            'train_both':self.train_both,
            'train_all':self.train_all,
            'resnet_path':self.resnet_path
        }

class Accuracy(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, logit:torch.Tensor, target:torch.Tensor):
        pred = torch.argmax(logit, dim=-1)
        correct_counts = torch.sum(pred==target)/pred.shape[0]
        all_correct = 1 if torch.sum(pred==target) == pred.shape[0] else 0
        return correct_counts.item(), all_correct

##参考ResNet18的实现
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*16, 2048)
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(2048, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = torch.squeeze(out)
        if len(out.shape) == 3:
            out = out.unsqueeze(0)
        out = out.reshape(out.shape[0], -1)
        try:
            out = self.linear(out)
        except:
            ic(out.shape)
        out = self.dropout(out)
        out = self.linear2(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])
## 参考ResNet18的实现

class myDataset(Dataset):
    def __init__(self, data_dir:str, labels, transform=None, args=None, train=True) -> None:
        super().__init__()
        self.transform = transform
        self.data_dir = data_dir
        self.meta_data = os.listdir(self.data_dir)
        self.labels = labels
        self.args = args
        self.is_train = train
          
    def __len__(self) -> int:
        return len(self.meta_data)

    def __getitem__(self, index: int):
        img_counts = len(os.listdir(self.data_dir+'/'+self.meta_data[index])) - 1
        imgs = []
        for i in range(img_counts):
            file_name = self.data_dir+'/'+self.meta_data[index]+'/'+self.meta_data[index]+'_'+str(i)+'.jpg'
            temp_img = Image.open(file_name)
            if self.transform is not None:
                temp_img = self.transform(temp_img)
            if self.args.train_resnet or self.args.train_both:
                imgs.append(temp_img.reshape(1,3,124,124))
            else:
                imgs.append(temp_img.reshape(1,3,224,224))
        opt_labels = deepcopy(self.labels[self.meta_data[index]]['optional_tags'])
        _gt_labels = deepcopy(self.labels[self.meta_data[index]]['imgs_tags'])
        if type(opt_labels) is not list:
            opt_labels = [opt_labels]
        if type(_gt_labels) is not list:
            _gt_labels = [_gt_labels]
        gt_labels = []
        if self.is_train:
            for item in _gt_labels:
                v = None
                for _,va in item.items():
                    v = va
                gt_labels.append(opt_labels.index(v))
        else:
            gt_labels = deepcopy(opt_labels)
        same_count = 0
        quit_flag = False
        for i in range(1,100):
            try:
                ch = opt_labels[0][-i]
            except:
                break
            try:
                for j in range(1, len(opt_labels)):
                    if not(opt_labels[j][-i] == ch):
                        quit_flag = True
                        break
            except:
                break
            if quit_flag:
                same_count = i - 1
                break
        for i in range(len(opt_labels)):
            color = opt_labels[i]
            if not(same_count == 0):
                color = color.rstrip(opt_labels[-1][-same_count:])
            color = color.replace('色','')
            opt_labels[i] = color
        if self.is_train:
            gt_labels = torch.tensor(gt_labels)
            return imgs, opt_labels, gt_labels
        else:
            return imgs, opt_labels, gt_labels, self.meta_data[index]

class trainer:
    def __init__(self, args):
        self.args = args
        self.init_net = InitNet(args)
        self.init_net.eval()
        self.net = MatchingNet(args)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr)
        self.metric = Accuracy()
        self.loss = torch.nn.CrossEntropyLoss()
        self.epoch = self.args.epoch
        if args.inference is None:
            wandb.init(project='course_project', config=self.args.config)
        self.step = 0
        self.val_step = 0
        if self.args.checkpoint_path is not None:
            state_dict = torch.load(self.args.checkpoint_path)
            self.net.load_state_dict(state_dict)
            print('Successfully loaded checkpoints from {}'.format(self.args.checkpoint_path))
        if self.args.resnet_path is not None:
            state_dict = torch.load(self.args.resnet_path)
            self.init_net.resnet.load_state_dict(state_dict)
            print('Successfully loaded checkpoints from {}'.format(self.args.resnet_path))
        if self.args.cuda:
            self.net.to('cuda')
            self.init_net.to('cuda')
        if args.train_resnet and not args.fix_resnet:
            self.contrastive_loss = SupConLoss(args)

    def read_data(self):
        with open(self.args.path+'/test_all.json') as f:
	        self.test_text = json.load(f)
        
        with open(self.args.path+'/train_all.json') as f:
            train_text = json.load(f)
        
        if self.args.crop:
            train_trans = transforms.Compose([
                transforms.Resize([224,224]),
                transforms.CenterCrop((200, 200)),
                transforms.Resize([224,224]) if (not self.args.train_resnet) and (not self.args.train_both) else transforms.Resize([124,124]),
                transforms.ToTensor(),
            ])
            test_trans = transforms.Compose([
                transforms.Resize([224,224]),
                transforms.CenterCrop((200, 200)),
                transforms.Resize([224,224]) if (not self.args.train_resnet) and (not self.args.train_both) else transforms.Resize([124,124]),
                transforms.ToTensor(),
            ])
        elif self.args.extra_crop:
            train_trans = transforms.Compose([
                transforms.Resize([224,224]),
                transforms.CenterCrop((200, 200)),
                transforms.Resize([224,224]) if (not self.args.train_resnet) and (not self.args.train_both) else transforms.Resize([124,124]),
                #transforms.RandomRotation(180),
                transforms.GaussianBlur(11, 3),
                transforms.ToTensor(),
            ])
            test_trans = transforms.Compose([
                transforms.Resize([224,224]),
                transforms.CenterCrop((200, 200)),
                transforms.Resize([224,224]) if (not self.args.train_resnet) and (not self.args.train_both) else transforms.Resize([124,124]),
                transforms.GaussianBlur(11, 3),
                transforms.ToTensor(),
            ])
        else:
            train_trans = transforms.Compose([
                transforms.Resize([224,224]),
                transforms.ToTensor(),
            ])
            test_trans = transforms.Compose([
                transforms.Resize([224,224]),
                transforms.ToTensor(),
            ])
        train_set = myDataset(data_dir=self.args.path+'/train', labels=train_text, transform=train_trans, args=args)
        self.test_set = myDataset(data_dir=self.args.path+'/test', labels=self.test_text, transform=test_trans, args=args, train=False)
        if self.args.train_all:
            self.train_set = train_set
        else:
            self.train_set, self.valid_set = random_split(train_set, [int(0.8 * len(train_set)), len(train_set) - int(0.8 * len(train_set))])
        self.idx_list = list(range(len(self.train_set)))

    def valid_test(self):
        self.net.eval()
        self.init_net.init_eval()
        with torch.no_grad():
            total_acc = []
            all_acc = []
            for imgs, word_vec, labels in self.valid_set:
                imgs, word_vec = self.init_net(imgs, word_vec)
                output = self.net(imgs, word_vec).cpu()
                acc, all = self.metric(output, labels)
                total_acc.append(acc)
                all_acc.append(all)
            
            total_acc = np.mean(total_acc)
            all_acc = np.mean(all_acc)
            wandb.log({"val_total_acc":total_acc, 'val_full_acc':all_acc}, step = self.val_step)
            self.val_step += 1
            print(f"Avg {self.metric}: {total_acc:.4f}")
            print(f"Avg {self.metric}: {all_acc:.4f}")

        return total_acc, all_acc

    def inference(self):
        self.init_net.eval()
        self.init_net.init_eval()
        self.net.eval()
        print('start inference...')
        idx = 0
        for imgs, labels, gt, meta in self.test_set:
            idx += 1
            if idx % 500 == 0:
                print('done '+str(idx))
            imgs, word_vec = self.init_net(imgs, labels)
            output = self.net(imgs, word_vec)
            pred = torch.argmax(output, dim=-1)
            dic = self.test_text[meta]['imgs_tags']
            for i in range(len(pred)):
                ans = gt[pred[i]]
                temp_dic = dic[i]
                for k,v in temp_dic.items():
                    dic[i][k] = ans
            self.test_text[meta]['imgs_tags'] = dic

        with open(self.args.inference+'_ans.json', 'w', encoding='utf-8') as json_file:
            json.dump(self.test_text, json_file, ensure_ascii=False)
            print("write json file success!")
            


    def train(self):
        for epoch in range(self.epoch):
            self.net.train()
            if self.args.train_resnet and not self.args.fix_resnet:
                self.init_net.init_train()
                self.train_resnet_epoch(epoch)
                torch.save(self.init_net.resnet.state_dict(), wandb.run.dir+'/resnet_epoch{}.pth'.format(epoch))
            else:
                if self.args.train_both:
                    self.init_net.init_train()
                self.train_epoch(epoch)
                if not self.args.train_all:
                    self.valid_test()
                torch.save(self.net.state_dict(), wandb.run.dir+'/epoch{}.pth'.format(epoch))
                if self.args.train_both:
                    torch.save(self.init_net.resnet.state_dict(), wandb.run.dir+'/resnet_epoch{}.pth'.format(epoch))

    def train_resnet_epoch(self, epoch):
        self.init_net.resnet.train()
        random.shuffle(self.idx_list)
        idx = 0
        imgs_ = []
        labels_ = []
        while idx < len(self.idx_list):
            idx_ = self.idx_list[idx]
            imgs, _, labels = self.train_set[idx_]
            imgs_ = imgs_ + imgs
            if len(labels_) > 0:
                max_idx = torch.max(torch.cat(labels_))
            else:
                max_idx = -1
            labels = max_idx + labels + 1
            labels_.append(labels)
            idx += 1

            if (idx+1) % self.args.batch_size == 0:
                imgs = torch.cat(imgs_, dim=0)
                labels = torch.cat(labels_)
                if self.args.cuda:
                    imgs = imgs.cuda()
                    labels = labels.cuda()
                imgs = self.init_net.resnet(imgs)
                total_loss = self.contrastive_loss(imgs, labels)
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                wandb.log({'loss':total_loss.item()}, step = self.step)
                self.step += 1
                imgs_ = []
                labels_ = []
                print(f"epoch:{epoch} - batch:{self.step}, loss:{total_loss.item():.4f}")
    

    def train_epoch(self, epoch):
        random.shuffle(self.idx_list)
        idx = 0
        batch_counts = 0
        loss = []
        acc_ = []
        full_acc = []
        while idx < len(self.idx_list):
            idx_ = self.idx_list[idx]
            imgs, word_vec, labels = self.train_set[idx_]
            imgs, word_vec = self.init_net(imgs, word_vec)
            output = self.net(imgs, word_vec)
            idx += 1
            if self.args.cuda:
                labels = labels.cuda()
            local_loss = self.loss(output, labels)
            loss.append(local_loss)
            acc, all = self.metric(output.cpu(), labels.cpu())
            acc_.append(acc)
            full_acc.append(all)

            if (idx+1) % self.args.batch_size == 0:
                total_loss = loss[0]
                for i in range(1,len(loss)):
                    total_loss = total_loss + loss[i]
                total_loss = total_loss/len(loss)
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                batch_counts += 1
                loss = []
                wandb.log({'training loss':total_loss.item(),'training_acc':np.mean(acc_), 'training_all_acc':np.mean(full_acc)}, step = self.step)
                self.step += 1
                acc_ = []
                full_acc = []
                print(f"epoch:{epoch} - batch:{batch_counts}, loss:{total_loss.item():.4f}")

## 参考监督对比学习代码
class SupConLoss(nn.Module):

    def __init__(self, args, temperature=0.1, scale_by_temperature=False):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature
        self.cuda = args.cuda

    def forward(self, features, labels):
        if self.cuda:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  # 计算两两样本间点乘相似度
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)    
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask
        
        num_positives_per_row  = torch.sum(positives_mask , axis=1) # 除了自己之外，正样本的个数  [2 0 2 2]       
        denominator = torch.sum(
        exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True)  
        
        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        

        log_probs = torch.sum(
            log_probs*positives_mask , axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]
        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss
## 参考监督对比学习代码

def init_(m):
    init_method = nn.init.orthogonal_
    gain = nn.init.calculate_gain('relu')
    return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

## 参考SuperGlue
def MLP(channels, do_bn=False):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)

class MLPAttention(nn.Module):
    def __init__(self, desc_dim):
        super().__init__()
        self.mlp = MLP([desc_dim * 2, desc_dim, 1])
        self.desc_dim = desc_dim

    def forward(self, query, key, value):
        
        nq, nk = query.size(-1), key.size(-1)
        
        scores = self.mlp(torch.cat((
            query.view(1, -1, nq, 1).repeat(1, 1, 1, nk).view(1, -1, nq * nk),
            key.view(1, -1, 1, nk).repeat(1, 1, nq, 1).view(1, -1, nq * nk)), dim=1)).view(1, nq, nk)    
        
        prob = scores.softmax(dim=-1)
        return torch.einsum('bnm,bdm->bdn', prob, value), scores
        
class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([(self.merge) for _ in range(3)])

    def attention(self, query, key, value):
        dim = query.shape[1]
        scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
        prob = torch.nn.functional.softmax(scores, dim=-1)
        return torch.einsum('bhnm,bdhm->bdhn', prob, value), scores

    def forward(self, query, key, value):
        batch = query.shape[0]
        query, key, value = [l(x).view(batch, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, scores = self.attention(query, key, value)
        return self.merge(x.contiguous().view(batch, self.dim*self.num_heads, -1)), scores.mean(1)

class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int, type: str):
        super().__init__()
        self.attn = MLPAttention(feature_dim) if type == 'cross' else MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message, weights = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1)), weights

class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.attn = nn.ModuleList([AttentionalPropagation(feature_dim, 4, type) for type in layer_names])
        self.names = layer_names

    def forward(self, imgs, word_vec):
        # imgs: n*hidden
        # word_vec: k*hidden
        desc0, desc1 = imgs, word_vec
        for attn, name in zip(self.attn, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:
                src0, src1 = desc0, desc1
            
            
            delta0, score0 = attn(desc0, src0)
            delta1, score1 = attn(desc1, src1)

            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)

        score = torch.nn.functional.softmax(score1,dim=-1)
        
        return score

## 参考SuperGlue

class InitNet(torch.nn.Module):
    def __init__(self, args):
        super(InitNet, self).__init__()       
        self.bertModel = BertModel.from_pretrained('bert-base-chinese')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        if (args.train_resnet and not args.fix_resnet) or args.train_both:
            self.resnet = ResNet18()
            self.resnet.train()
        elif not args.train_resnet:
            self.resnet = resnet18(pretrained=True)
            self.resnet.eval()
        else:
            self.resnet = ResNet18()
            self.resnet.eval()
        self.bertModel.eval()
        self.args = args
    
    def init_train(self):
        self.resnet.train()
    
    def init_eval(self):
        self.resnet.eval()
        
    def forward(self, imgs, opt_labels):
        with torch.no_grad():
            token = self.tokenizer(opt_labels, padding=True, truncation=True, max_length=5,return_tensors = 'pt')
            inp = token['input_ids'].cuda() if self.args.cuda else token['input_ids']
            outputs = self.bertModel(inp)
            word_vec = outputs['last_hidden_state']
            #word_vec = torch.mean(word_vec, dim=1)
        if (self.args.train_resnet and not self.args.fix_resnet) or self.args.train_both:
            imgs = torch.cat(imgs, axis=0)
            if len(imgs.shape) == 3:
                imgs = imgs.unsqueeze(0)
            if self.args.cuda:
                imgs = imgs.cuda()
            imgs = self.resnet(imgs)
        else:
            with torch.no_grad():
                imgs = torch.cat(imgs, axis=0)
                if self.args.cuda:
                    imgs = imgs.cuda()
                imgs = self.resnet(imgs)
        
        return imgs, word_vec

class MatchingNet(torch.nn.Module):
    def __init__(self, args):
        super(MatchingNet, self).__init__()       
        self.hidden_size = args.hidden_size
        self.cuda = args.cuda
        self.args = args
        self.word_gru = torch.nn.GRU(input_size=768, hidden_size=768, batch_first=True)
        self.vis_encoder = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, self.hidden_size)
        )

        self.word_encoder = nn.Sequential(
            nn.Linear(768,512),
            nn.ReLU(),
            nn.Linear(512,self.hidden_size)
        )

        feature_dim = self.args.hidden_size
        gnn_layers = 3 * ['self', 'cross']
        self.gnn = AttentionalGNN(feature_dim, gnn_layers)
        if self.args.use_attn:
            self.attn = torch.nn.MultiheadAttention(int(self.hidden_size/8),num_heads=2)
        
    def forward(self, imgs, word_vec):
        if self.args.normalize == 'batch':
            F.normalize(imgs, dim=0)
            F.normalize(word_vec, dim=0)
        elif self.args.normalize == 'sample':
            F.normalize(imgs, dim=-1)
            F.normalize(word_vec, dim=-1)
        if self.args.use_gru:
            word_vec = self.word_gru(word_vec)[0][:,-1]
        else:
            word_vec = torch.mean(word_vec, dim=1)
        word_vec = self.word_encoder(word_vec)

        imgs = self.vis_encoder(imgs)
        if self.args.use_attn:
            imgs = imgs.reshape(imgs.shape[0], 8, int(self.hidden_size/8)).transpose(1,0)
            out,_ = self.attn(imgs, imgs, imgs)
            out = out.transpose(1,0)
            imgs = out.reshape(imgs.shape[1],-1)
        imgs = imgs.transpose(0,1).unsqueeze(0)
        word_vec = word_vec.transpose(0,1).unsqueeze(0)
    
        e = self.gnn(word_vec, imgs)

        e = e.reshape(e.shape[1], e.shape[2])
        
        return e

if __name__ == '__main__':
    args = Args()
    train = trainer(args)
    train.read_data()
    if args.inference is not None:
        train.inference()
    else:
        train.train()