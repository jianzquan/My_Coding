import os, copy, pickle, random, torch, nltk, json
from torch.utils.data import random_split

import numpy as np

def wordTokenize(tokens): 
    return [token for token in nltk.word_tokenize(tokens)]

def iterSupport(func, query):
    # 迭代处理 list 数据
    if isinstance(query, (list, tuple)):
        return [iterSupport(func, q) for q in query]
    
    try:
        return func(query)
    except TypeError:
        return func[query]

def getSample(lines):
    context, aspect, label = lines[0].strip().lower(), lines[1].strip().lower(), int(lines[2])+1
    c_tokens = context.split(' ') # 上下文 分词
    a_tokens = aspect.split(' ')  # aspect 分词
    # 定位 aspect 位置
    c_position = []
    for wi, word in enumerate(c_tokens):
        if word == '$t$':
            w_start, w_end = wi, wi+len(a_tokens)
            c_tokens = c_tokens[:w_start] + ['<pad>']*len(a_tokens) + c_tokens[w_end:]
            s_tokens = c_tokens[:w_start] + a_tokens + c_tokens[w_end:]
            assert len(s_tokens) == len(c_tokens)
            c_position = [w_start-i if i<=w_start else i-w_end+1 for i in range(len(c_tokens))]
            c_position = [val if val>=0 else 0 for val in c_position]

            break

    return {'index': 0, 'sent': context, 'seq': s_tokens, 'seq_len': len(s_tokens), 'seq_pos': c_position, 'seq_atn': [], 'asp': a_tokens, 'asp_len': len(a_tokens), 'asp_pos': list(range(w_start, w_end+1)), 'label': label}

def getIndex(vocab, train=None, val=None, test=None):

    vocab.datas = {'train': train, 'val': val, 'test': test}
    for key, data_all in vocab.datas.items():
        if data_all is None: continue
        data, temp = [], np.zeros(vocab.maxSeqLen).astype('int32')
        for i, item in enumerate(data_all):
            item['index'] = i
            item['seq_id']   = [vocab.word2id[word] if word in vocab.word2id else 1 for word in item['seq']]
            item['seq_mask'] = [1 for _ in item['seq']]
            item['ctx_mask'] = [1 if i<item['asp_pos'][0] or i>=item['asp_pos'][-1] else 0 for i in range(item['seq_len'])]
            
            item['asp_mask'] = [1-val for val in item['ctx_mask']]
            item['asp_id']     = [vocab.word2id[word] if word in vocab.word2id else 1 for word in item['asp']]
            assert sum(item['asp_mask']) == item['asp_len']

            # padding
            samp = {'indexes': 0, 'labels': 0, 'asps_len': 0, 'seqs_len': 0, 'seqs_id': copy.deepcopy(temp), 'seqs_mask': copy.deepcopy(temp), 'asps_mask': copy.deepcopy(temp), 'ctxs_mask': copy.deepcopy(temp)}
            samp['indexes']  = item['index']
            samp['labels']   = item['label']
            samp['seqs_len'] = item['seq_len']
            samp['asps_len'] = item['asp_len']
            samp['seqs_id'][0:item['seq_len']]   = item['seq_id']
            samp['seqs_mask'][0:item['seq_len']] = item['seq_mask']
            samp['asps_mask'][0:item['seq_len']] = item['asp_mask']
            samp['ctxs_mask'][0:item['seq_len']] = item['ctx_mask']

            data.append(samp)
        
        if key == 'train': vocab.train = data
        if key == 'test':  vocab.test  = data
        if key in ['dev', 'val']: vocab.val = data
    
    if val is None and vocab.args.val_ratio>0: 
        val_len = int(len(vocab.train)*vocab.args.val_ratio)
        train_len = len(vocab.train) - val_len
        vocab.train, vocab.val = random_split(vocab.train, [train_len, val_len])



def process_sentence(question):
    if " '" in question: question = question.replace(" '", " ` ")
    if "' " in question: question = question.replace("' ", " ' ")
    if "'?" in question: question = question.replace("'?", " '?")
    if "'s" in question: question = question.replace("'s", " 's")
    if ': ' in question: question = question.replace(': ', ' : ')
    if '%' in question:  question = question.replace('%', ' %')
    if '$' in question:  question = question.replace('$', '$ ')
    if '=' in question:  question = question.replace('=', ' = ')
    if '- ' in question and '--' not in question: question = question.replace('- ', ' - ')
    if '?' in question and '??' not in question:  question = question.replace('?', ' ?')
    if '??' in question: question = question.replace('??', ' ??')
    if '. ' in question and 'u.s.' not in question and 'jr.' not in question and ' v.' not in question \
            and 'dec.' not in question and 'h.j.' not in question and 'c.w.' not in question \
            and 'st.' not in question and 'dr.' not in question and ' w.' not in question \
            and ' h.' not in question and ' r.' not in question and ' a.' not in question \
            and 'd.c.' not in question and ' d.' not in question and ' mt.' not in question \
            and 'mr.' not in question and 'f.c.' not in question and ' j.' not in question \
            and 'u.e.' not in question and ' c.' not in question and ' m.' not in question \
            and ' b.' not in question and ' no.' not in question and ' op.' not in question \
            and 'rev.' not in question and 'u.k.' not in question and 'l.a.' not in question \
            and 'mrs.' not in question and 'm.sc' not in question and ' f.' not in question \
            and 'inc.' not in question and ' e.' not in question and 'g.m.c.' not in question:
        question = question.replace('. ', ' . ')
    if '(' in question and ')' in question: 
        question = question.replace('(', '-lrb- ')
        question = question.replace(')', ' -rrb-')
    if '[' in question and ']' in question:
        question = question.replace('[', '-lsb- ')
        question = question.replace(']', ' -rsb-')
    if '\u27e8' in question and '\u27e9':
        question = question.replace('\u27e8', '\u27e8 ')
        question = question.replace('\u27e9', ' \u27e9')
    if '\u00a3' in question:
        question = question.replace('\u00a3', '# ')
    if '\u2013' in question and ' \u2013 ' not in question:
        question = question.replace('\u2013', ' -- ')
    if "n't" in question: question = question.replace("n't", " n't")
    if "'re" in question and " 're" not in question: question = question.replace("'re", " 're")
    if "'ve" in question: question = question.replace("'ve", " 've")
    while '\"' in question:
        if '\"' in question:
            index = question.find('\"')
            question = question[:index] + '`` ' + question[index + 1:]
        if '\"' in question:
            index = question.find('\"')
            question = question[:index] + " ''" + question[index + 1:]
    if ', ' in question: question = question.replace(', ', ' , ')
    if question[-1] == '.' or question[-1] == '>' or question[-1] == '/':
        question = question[:-1] + ' ' + question[-1]
    question = question.replace('   ', ' ')
    question = question.replace('  ', ' ')
    return question

def parseData(vocab, path):

    train, val, test = None, None, None
    for key, path in path.items():
        with open(path, 'r', newline='\n', errors='ignore') as f:
            json_data = json.load(f) # 398项(title+paragraphs)，若干paragraph(context+qas), 若干qas(answers+question) 
            qa_dict = {}
            for item_t in json_data:
                title, paragraphs = item_t['title'], item_t['paragraphs']
                for item_p in paragraphs:
                    context, qas = item_p['context'], item_p['qas']
                    for item_q in qas:
                        question, answer = item_q['question'].strip().lower(), item_q['answers'][0]
                        answer_id, answer_text = answer['answer_start'], answer['text'].strip().lower()
                        question = process_sentence(question)
                        answer_text = process_sentence(answer_text)
                        qa_dict[question] = answer_text
                        



                        pass

                    pass
                pass

                if i >= len(lines) * vocab.ratio: break
                sample_raw = lines[i:i+3]
                sample = getSample(sample_raw)
                data.append(sample)
                vocab.maxSeqLen = max(vocab.maxSeqLen, sample['seq_len'])
                vocab.maxAspLen = max(vocab.maxAspLen, sample['asp_len'])
                vocab.Distri[key][sample['label']] += 1
        vocab.labels = list(set([item['label'] for item in data]))
        vocab.n_class = len(vocab.labels)

        if key == 'train': 
            train = data
            # 构建 word2id, 获取 word 字典对应的词向量
            if vocab.args.model_name == 'bert':
                vocab.word_embed = []
            else:
                words = [item['seq'] for item in data]
                iterSupport(vocab._add, words)
                if vocab.args.glove_dir != 'none': 
                    pretrainEmbed(vocab, desc=vocab.args.embed_type)
        if key in ['val', 'dev']: val = data
        if key == 'test': test = data

    getIndex(vocab, train, val, test)

def pretrainEmbed(vocab, desc='glove'):

    if desc == 'glove': path = vocab.args.glove_dir

    # 1. 导入 Embedding Vectors
    if os.path.exists(path + '.cache'):
        with open(path+'.cache', 'rb') as fr:
            embed_matrix = pickle.load(fr) 
    else:
        embed_matrix = {}
        with open(path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            for line in f:
                tokens = line.rstrip().split()
                word   = ''.join(tokens[0:len(tokens)-vocab.embed_dim])
                embed_matrix[word] = [float(ele) for ele in tokens[-vocab.embed_dim:]]
            # 保存缓存文件
            with open(path+'.cache', 'wb') as fw:
                pickle.dump(embed_matrix, fw)

    # 2. 匹配字典
    Matrix = [] # 每行表示一个 embedding vector
    dim = len(list(embed_matrix.values())[0]) # 获取向量维度
    for key, value in vocab.word2id.items():
        if key == "<pad>":
            Matrix.append([0] * dim)
        elif key in embed_matrix:
            Matrix.append([float(ele) for ele in embed_matrix[key]])
        else:
            Matrix.append([random.uniform(0, 1) for i in range(dim)]) # 随机生成
        assert len(Matrix) == value+1
        
    vocab.word_embed = Matrix
    print("***** local word matrix is loaded ! *****")


class SQuAD():

    def __init__(self, args, ratio=1):

        self.args      = args
        self.name      = args.data_name
        self.ratio     = ratio  # 处理数据的比例 
        self.word2id   = {'<pad>': 0, '<unk>': 1}
        self.id2word   = {0: '<pad>', 1: '<unk>'}
        self.wordCount = {'<pad>': 0, '<unk>': 1}
        self.embed_dim = args.embed_dim
        self.labels    = []  # 分类类别 []
        self.maxSeqLen = 0   # 最长 context 长度
        self.maxAspLen = 0   # 最长 aspect 长度
        self.Distri    = {'train': [0,0,0], 'test': [0,0,0]}
        # self.tokenizer = BertTokenizer.from_pretrained(args.pretrain)

        if self.name=='SQuAD' or self.name=='SQuAD-1.1':
            filePath = {'train': args.data_dir + "SQuAD-1.1/train.json",
                        'val':   args.data_dir + "SQuAD-1.1/dev.json",
                        'test' : args.data_dir + "SQuAD-1.1/test.json"}
        else: print('no data file found !'); return -1

        parseData(self, filePath)
        pickle.dump(self, open(args.data_dir+'{}/Data'.format(self.name), 'wb'))

    def _add(self, ele):
        # 添加新词, 统计旧词
        if ele not in self.wordCount:
            self.word2id[ele] = len(self.word2id)
            self.id2word[len(self.id2word)] = ele
            self.wordCount[ele] = 0
        self.wordCount[ele] += 1


        


