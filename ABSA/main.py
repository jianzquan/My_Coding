import torch, logging, os, time, pickle, warnings
from numpy import random
from Models.TNet import TNet
from Models.MemN2N import MemN2N
from Models.MyBert import MyBert
from Models.AS import Train_AS
from Models.AT import Train_AT
from utils.aids import fix_random_state, Config
warnings.filterwarnings("ignore")
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

# def clearDirs(args):
#     # 0. outs; 1. runs; 2. log
#     if os.path.exists(args.output_dir): 
#         for root, dirs, files in os.walk(args.output_dir, topdown=False):
#             for file in files: os.remove(os.path.join(root, file)) 
#             # for dir in dirs: os.remove(os.path.join(root, dir))
    
#     if os.path.exists(args.tb_dir): 
#         for root, dirs, files in os.walk(args.tb_dir, topdown=False):
#             for file in files: os.remove(os.path.join(root, file)) 


# 一个数据集、一个模型、一次寻优
def main(args):

    begin_time = time.time()
    # 0. import Data 
    dataFile = args.data_dir+args.data_name+'/Data'
    if os.path.exists(dataFile): Data = pickle.load(open(dataFile, 'rb'))
    else: from utils.DataLoad import ABSA; Data = ABSA(args, ratio=1)

    # 1. import Model and train
    model = args.models[args.model_name](args, Data).to(args.device)
    from utils.Processor import Processor, initial_params
    initial_params(model); processor = Processor(args, Data, model)
    if args.model_method=='no': 
        processor.train(desc='train')
    if args.model_method=='au': 
        Train_AT(processor)
    if args.model_method in ['sm', 'sa', 'as']:
        Train_AS(processor)

    # 2. load Model and test
    processor.model = torch.load(args.output_dir)
    score_dev, score_test = processor.test(desc='test')
    end_time = time.time()
    print('>>> dev_score: {:.2f}, test_score: {:.2f}, time: {:.2f} \n'.format(score_dev, score_test, end_time-begin_time))


if __name__ == "__main__":

    # 0. parameters setting and seed fix
    args   = Config()
    logger = logging.getLogger(__name__)
    args.models   = {'MemN2N': MemN2N, 'TNet': TNet, 'Bert': MyBert}
    args.datasets = ['Rest', 'Laptop', 'Twitter']
    args.methods  = ['no', 'sm', 'sa', 'as', 'au']

    dataRes, modelRes = {}, {} # save all results

    # 1. parameters selection
    args.model_method = args.methods[0] 
    args.model_name   = list(args.models.keys())[0]
    args.data_name    = args.datasets[0]

    # 2. optimization
    args.score_dev, args.score_test = 0, 0
    name = "{}_{}_{}".format(args.data_name, args.model_name, args.model_method)
    args.output_dir += "{0}/outs/{1}".format(args.data_name, name)
    args.base_model = 0 # 每次都从头开始跑
    for i in range(1):
        args.seed = i+2029; fix_random_state(args.seed)
        params = {'lr': [0.1, 0.05, 0.01], 'batch_size': [8, 16, 32], 'drop_rate': [0.1, 0.3]}
        args.lr = random.choice(params['lr'])
        args.drop_rate = random.choice(params['drop_rate'])
        args.batch_size = int(random.choice(params['batch_size']))
        main(args)
