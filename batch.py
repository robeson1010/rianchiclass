#hide
from RS.core import *
from RS.model import *

from fastai.vision.all import *
from mmseg.models import build_segmentor
from mmcv.utils import Config
from fastai.distributed import *
# df=pd.read_csv('./train.csv')
imgs=get_files('/home/staff/xin/Downloads/tianchi/train/',extensions='.tif')
df=pd.DataFrame({'path':imgs})

db = DataBlock(blocks=(TransformBlock(type_tfms=partial(RsImage.create)),
                       TransformBlock(type_tfms=partial(RsMask.create)),
                      ),
               get_x=ColReader('path'),
               get_y=ColReader('path'),
               splitter=RandomSplitter(valid_pct=0.2,seed=10),
               item_tfms=[aug,aug2]
              )

dls = db.dataloaders(source=df,bs=64, num_workers=12)
class HRNET(nn.Module):
    def __init__(self,cfgfile):
        super().__init__()
        cfg = Config.fromfile(cfgfile)
        temp=build_segmentor(cfg.model)
        self.backbone =temp.backbone
        self.decode_head = temp.decode_head
    def forward(self, x):
        x = self.backbone(x)
        x = self.decode_head(x)
        return x

model=HRNET(cfgfile='./fcn_hr18.py')
apply_init(model)

def mIOU( pred,label, num_classes=10):
    pred =F.upsample_nearest(pred,scale_factor=4)  
    pred = F.softmax(pred, dim=1)              
    pred = torch.argmax(pred, dim=1).squeeze(1)
    
    iou_list = list()
    present_iou_list = list()
    pred = pred.view(-1)
    label = label.view(-1)
    # Note: Following for loop goes from 0 to (num_classes-1)
    # and ignore_index is num_classes, thus ignore_index is
    # not considered in computation of IoU.
    for sem_class in range(num_classes):
        if sem_class!=30:
            pred_inds = (pred == sem_class)
            target_inds = (label == sem_class)
            if target_inds.long().sum().item() == 0:
                iou_now = float('nan')
            else: 
                intersection_now = (pred_inds[target_inds]).long().sum().item()
                union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
                iou_now = float(intersection_now) / float(union_now)
                present_iou_list.append(iou_now)
            iou_list.append(iou_now)
    return np.mean(present_iou_list)

class OhemCrossEntropy(nn.Module): 
    def __init__(self, ignore_label=-1, thres=0.7, 
        min_kept=100000, weight=None): 
        super(OhemCrossEntropy, self).__init__() 
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label 
        self.weight=weight
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label, 
                                             reduction='none') 
    
    def forward(self, score, target, **kwargs):
        target=RsImage(target.long())
        score =F.upsample_nearest(score,scale_factor=4)
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label         
        
        tmp_target = target.clone() 
        tmp_target[tmp_target == self.ignore_label] = 0 
        pred = pred.gather(1, tmp_target.unsqueeze(1)) 
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)] 
        threshold = max(min_value, self.thresh) 
        
        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold] 
#         iouloss=Lov.lovasz_softmax(score,target,class_weight=self.weight)
        return pixel_losses.mean()

weight=torch.FloatTensor([2,1,4,4,2,2,2,4,2,4]).cuda()
loss=OhemCrossEntropy(weight=weight)

learn = Learner(dls,model,metrics=mIOU,loss_func=loss).to_fp16()

learn.load('hrnet')
with learn.distrib_ctx():learn.fit_one_cycle(300, 1e-3,cbs=[CSVLogger(fname='hrnet.csv',append=True),SaveModelCallback(monitor='mIOU',fname='hrnet')])