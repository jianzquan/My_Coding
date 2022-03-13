
import torch, copy, pickle, os
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.Processor import Processor, initial_params
e = torch.tensor(1e-32)


def a_loss(self, batch, pred, attn):
    indexes = batch['indexes']
    keys = [self.as_Data['key'][i] for i in indexes]
    vals = [self.as_Data['val'][i] for i in indexes]

    max_len = attn.shape[-1]
    for i in range(len(indexes)):
        keys[i] = keys[i][0:max_len]
        vals[i] = vals[i][0:max_len]
    keys, vals = torch.tensor(keys), torch.tensor(vals)

    mask_sm = keys < 0
    loss_sm = torch.sum((attn*mask_sm - vals*mask_sm)**2)

    mask_sa = keys > 0
    loss_sa = torch.sum((attn*mask_sa - vals*mask_sa)**2)

    if   self.model.method == 'sm': loss = loss_sm /len(indexes) 
    elif self.model.method == 'sa': loss = loss_sa /len(indexes) 
    elif self.model.method == 'as': loss = (loss_sm+loss_sa) /len(indexes) 
    else: return 0

    return loss * 0.1


def as_Train(processor, k_epochs=5, threshold=3.0):
    args = processor.args
    Data = pickle.load(open(args.data_dir+args.data_name+'\\Data', 'rb'))
    data = Data.train.dataset
    train_indexes = Data.train.indices
    val_indexes = Data.val.indices
    temp = np.zeros_like(data[0]['seqs_mask'])
    as_Data = [copy.deepcopy(temp) for _ in data]
    for i in range(k_epochs):
        # 0. initial model
        model = args.models[args.model_name](args, Data).to(args.device)
        initial_params(model); self = Processor(args, Data, model)
        
        # training
        self.model.base_type = 0
        while self.early_stop <= args.early_stop:
            loss_train = self.oneEpoch()
        print('as_Train epoch_{} training is over ! \n'.format(i))
       
        # 寻找第i次重要词
        model = torch.load(args.output_dir)
        dataLoader = DataLoader(self.data.train, len(self.data.train.indices))
        for batch in dataLoader:
            model.eval()
            batch   = self.squeezePadding(batch)
            preds, attns = model(batch)
            preds_label = torch.argmax(F.log_softmax(preds, dim=-1), dim=-1)
            labels = batch['labels']

            for i in range(labels.shape[-1]):
                index = batch['indexes'][i]
                entropy = torch.sum(torch.cat([-val*torch.log(val+e) for val in attns[i].unsqueeze(dim=1)]))

                if entropy < threshold:
                    idx = attns[i].argmax()
                    if sum(data[index]['sent_mask'])==0:
                        continue
                    else:
                        data[index]['sent_mask'][idx] = 0 # 删词
                    if preds_label[i] == labels[i]:
                        as_Data[index][idx] = 1
                    else:
                        as_Data[index][idx] = -1
        del model
        
    # 重要词找完后进行处理
    as_Data_key, as_Data_val = [], []
    for key in as_Data:
        key, val = np.array(key), np.array(key)
        val = np.array([1/(key>0).sum() if i>0 else 0 for i in key])
        as_Data_key.append(key)
        as_Data_val.append(val)
    
    return {'key': as_Data_key, 'val': as_Data_val}


def Train_AS(self, desc='train'):

    # 统计数据
    args = self.args
    name = "as_Data_{}_{}_lr_{}_bz_{}_dr_{}".format(args.data_name,args.model_name,args.lr,args.batch_size,args.drop_rate)
    file = args.data_dir+args.data_name+"\\"+name
    if os.path.exists(file):
        as_Data = pickle.load(open(file, 'rb'))
    else:
        as_Data = as_Train(self, k_epochs=5, threshold=3.0)
        pickle.dump(as_Data, open(file, 'wb')) 

    self.as_Data = as_Data
    args, data, model = self.args, self.data, self.model
    print(">>> {}_{}, lr_{}, batch_size_{}".format(data.name, model.name+'_'+model.method, args.lr, args.batch_size))
    
    self.importModel()
    lossRec = np.array([])
    while self.early_stop <= self.args.early_stop: 
        loss_train = self.oneEpoch(other=a_loss)
        lossRec = np.append(lossRec, loss_train)