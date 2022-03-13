import argparse, random, torch, pickle
import numpy as np
from transformers import BertTokenizer


def fix_random_state(seed):

    random.seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)


def Config():

    # 参数设置
    argParse = argparse.ArgumentParser()  # 实例化对象

    # 0. 数据相关参数Rest
    argParse.add_argument('--val_ratio', type=float, default=0.0, 
                          help='val split from train')
    argParse.add_argument('--data_dir', type=str, default="../../Datasets/NLG/", 
                          help='directory of data, determine by input later')
    argParse.add_argument('--data_name', type=str, default="SQuAD", 
                          choices=['SQuAD',], 
                          help='class of NLG dataset')
    argParse.add_argument('--score_base', type=int, default=00.00, 
                          help='for saving base model')
    argParse.add_argument('--base_model', type=int, default=0, 
                          choices=[0, 1, 2, 3],
                          help='0: no use; 1: fix base model (=score_base); 2: fix base model (>score_base); 3: update base model')       
    argParse.add_argument('--n_class', type=int, default=None, 
                          help='label classes, re-defined when Data generation') 
    argParse.add_argument('--glove_dir', type=str, default="../../Glove/glove.840B/",  
                          help='directory of glove embeddings, select dimension later')
    
    # 1. 模型相关参数
    argParse.add_argument('--pretrain', type=str, default="bert-base-uncased",  
                          help='pretrained BERT large path')
    argParse.add_argument('--model_name', type=str, default='MemN2N',
                          choices=['PathQG',])           
    argParse.add_argument('--embed_type', type=str, default='glove',
                          choices=['none', 'glove', 'bert'])  
    argParse.add_argument('--embed_dim', type=int, default=300,
                          choices=[50, 100, 200, 300]) 
    argParse.add_argument('--drop_rate', type=float, default=0.1, 
                          choices=[0.1, 0.2, 0.3, 0.5, 0.8])
    argParse.add_argument('--batch_size', type=int, default=16, 
                          choices=[8, 16, 32, 64, 128])  # #

    # 2. 训练相关参数
    argParse.add_argument('--output_dir', type=str, default="../../Datasets/NLG/",  
                          help='determine when model initialization')
    argParse.add_argument('--tb_dir', type=str, default="../../Results/runs/",  
                          help='file dir for tensorboard.')
    argParse.add_argument('--seed', type=int, default=2022, 
                          help='random seed for result reproduce')                                
    argParse.add_argument('--device', type=str, default="cpu", 
                          choices=['cpu', 'cuda'])
    argParse.add_argument('--lr', type=float, default=0.01, 
                          help='initial learning rate for optimizer')  # #
    argParse.add_argument('--weight_decay', type=float, default=0, 
                          help='weight decay 0')  # 尚未了解
    argParse.add_argument('--momentum', type=float, default=0, 
                          help='momentum 0')  # 尚不了解
    argParse.add_argument('--max_grad_norm', type=float, default=1.0, 
                          help='max gradient norm')  # 尚不了解   
    argParse.add_argument('--epochs', type=int, default=30, 
                          help='Total number of training epochs')
    argParse.add_argument('--early_stop', type=int, default=10, 
                          help='if > 0: override epochs, use early stop strategy')
                          
    args = argParse.parse_args()  # 生成算法参数

    # 3. 相关参数修正
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('cuda is not available, cpu is working !!')
        args.device = "cpu"
    args.glove_dir = args.glove_dir+"glove.840B."+str(args.embed_dim)+"d.txt"

    if args.embed_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained(args.pretrain)
        args.tokenizer = tokenizer

    return args


def saveLoad(target, path, desc1, desc2):
    
    if desc2 == 'model':
        # 直接保存模型和加载模型 (加载模型的结构和参数)
        if desc1 == 'save':
            torch.save(target, path)
            # torch.save(target.state_dict(), path)
        if desc1 == 'load':
            return torch.load(path) 
            # target.load_state_dict(torch.load(path))

    if desc2 == 'variable':
        if desc1 == 'save':
            with open(path, 'wb') as fw:
                pickle.dump(target, fw)
        if desc1 == 'load':
            with open(path, 'rb') as fr:
                return pickle.load(fr)

    return -1