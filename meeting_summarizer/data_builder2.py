from transformers import BertTokenizer
import csv
import glob, os
import pandas as pd 
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

class BertDataProcessor:
    def __init__(self, data_dir, out_dir):
        self.data_dir = BASE_DIR+data_dir
        self.out_dir = BASE_DIR+out_dir
        self.inp_df = None
        self.meetings = set()

        for inp_file in sorted(glob.glob(self.data_dir+"/*txt")):
            f_name = os.path.basename(inp_file)
            self.meetings.add(f_name.split(".")[0])
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def split(self):
        '''
        train, validation and test split by meeting
        '''
        meetings = list(self.meetings)
#        self.train_list = meetings[:50]
#        self.validation_list = meetings[50:58]
#        self.test_list = meetings[58:]

        self.train_list = meetings[:10]
        self.validation_list = meetings[10:15]
        self.test_list = meetings[15:20]



    def format_to_bert(self, args=None):

        cls_vid = tokenizer.vocab["[CLS]"]
        sep_vid = tokenizer.vocab["[SEP]"]

        for batch, f_names in (("train", self.train_list), ("validation",self.test_list),
                 ("test", self.validation_list)):
            output = []
            for f_name in f_names:
                # process a meeting at a time
                pth = self.data_dir+"/"+f_name+"."+"*txt"
                cur_chunk = None
                cur_labels = []
                for inp_file in sorted(glob.glob(pth)):
                    with open(inp_file, "r") as mtg_f:
                        for line in mtg_f:
                            sent, label = line.split("|")
                            s_chunk = tokenizer.tokenize(sent) [:510]
                            s_chunk = ["[CLS]"] + s_chunk + ["[SEP]"]
                            if cur_chunk is None:
                                cur_chunk = s_chunk
                                cur_labels.append(int(label))

                            else:
                                # if new line fits in to remaining space, add it, else fill with spaces and add a new line
                                if len(s_chunk) + len(cur_chunk) < 512:
                                    cur_chunk += s_chunk
                                    cur_labels.append(int(label))
                                else:
                                    input_ids = tokenizer.convert_tokens_to_ids(cur_chunk)
                                    attn_masks = [1]*len(input_ids)
                                    cls_ids = [i for i, t in enumerate(input_ids) if t == cls_vid ]
                                    mask_cls = [1 for _ in range(len(cls_ids))]

                                    [attn_masks.append(0) for _ in range(len(attn_masks), 512)]
                                    [input_ids.append(0) for _ in range(len(input_ids), 512)]
                                    [cls_ids.append(0) for _ in range(len(cls_ids), 512)]
                                    [mask_cls.append(0) for _ in range(len(mask_cls), 512)]

                                    _segs = [-1] + [i for i, t in enumerate(input_ids) if t == sep_vid]
                                    segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
                                    segments_ids = []
                                    for i, s in enumerate(segs):
                                        if (i % 2 == 0):
                                            segments_ids += s * [0]
                                        else:
                                            segments_ids += s * [1]
                                    [cur_labels.append(0) for _ in range(len(cur_labels), 512)]
                                    [segments_ids.append(0) for _ in range(len(segments_ids), 512)]
                                    b_data_dict = {"src": input_ids, "labels": cur_labels, "segs": segments_ids, 
                                                'clss': cls_ids, "attn": attn_masks, "mask_cls":mask_cls}
                                    output.append(b_data_dict)
                                    cur_chunk = s_chunk
                                    cur_labels = [int(label)]
            out =  {"src": [], "labels": [], "segs": [], 
                                                'clss': [], "attn": [], "mask_cls":[]}
            for sample in output:
                for key, val in sample.items():
                    out[key].append(val)
            for k, v in out.items():
                out[k] = torch.LongTensor(v)
            for k, v in out.items():
                torch.save(v, self.out_dir+"/"+k+"_"+batch+".pt")
BASE_DIR = "/home/rajivn/W266/W266-fall-2020-hwu-rnair/"
dp = BertDataProcessor("data/ICSI_plus_NXT/processing", 
                    "data/ICSI_plus_NXT/tensors")
dp.split()
dp.format_to_bert()