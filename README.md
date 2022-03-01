# LIC2021关系抽取基线

信息抽取旨在从非结构化自然语言文本中提取结构化知识，如实体、关系、事件等。关系抽取的目标是对于给定的自然语言句子，根据预先定义的schema集合，抽取出所有满足schema约束的SPO三元组。schema定义了关系P以及其对应的主体S和客体O的类别。
本基线系统基于预训练语言模型ERNIE设计了结构化的标注策略，可以实现多条、交叠的SPO抽取。


## 环境依赖

* PaddlePaddle 安装

   本项目依赖于 PaddlePaddle 2.0 及以上版本，请参考 [安装指南](http://www.paddlepaddle.org/#quick-start) 进行安装

* PaddleNLP 安装

   ```shell
   pip install --upgrade paddlenlp\>=2.0.0rc5
   ```

* 环境依赖

   Python的版本要求 3.6+，其它环境请参考 PaddlePaddle [安装说明](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/beginners_guide/install/index_cn.html) 部分的内容


## 目录结构


以下是本项目主要目录结构及说明：

```text
event_extraction/
├── data_loader.py # 加载数据
├── extract_chinese_and_punct.py # 文本数据预处理
├── README.md # 文档说明
├── re_official_evaluation.py # 比赛评价脚本
├── run_duie.py # 模型训练脚本
├── train.sh # 启动脚本
└── utils.py # 效能函数
```

## 关系抽取基线

针对 DuIE2.0 任务中多条、交叠SPO这一抽取目标，比赛对标准的 'BIO' 标注进行了扩展。
对于每个 token，根据其在实体span中的位置（包括B、I、O三种），我们为其打上三类标签，并且根据其所参与构建的predicate种类，将 B 标签进一步区分。给定 schema 集合，对于 N 种不同 predicate，以及头实体/尾实体两种情况，我们设计对应的共 2*N 种 B 标签，再合并 I 和 O 标签，故每个 token 一共有 (2*N+2) 个标签，如下图所示。

<div align="center">
<img src="images/tagging_strategy.png" width="500" height="400" alt="标注策略" align=center />
</div>

### 评价方法

对测试集上参评系统输出的SPO结果和人工标注的SPO结果进行精准匹配，采用F1值作为评价指标。注意，对于复杂O值类型的SPO，必须所有槽位都精确匹配才认为该SPO抽取正确。针对部分文本中存在实体别名的问题，使用百度知识图谱的别名词典来辅助评测。F1值的计算方式如下：

F1 = (2 * P * R) / (P + R)，其中

- P = 测试集所有句子中预测正确的SPO个数 / 测试集所有句子中预测出的SPO个数
- R = 测试集所有句子中预测正确的SPO个数 / 测试集所有句子中人工标注的SPO个数

### 1：构建模型

该任务可以看作一个序列标注任务，所以基线模型采用的是ERNIE序列标注模型。

**PaddleNLP提供了ERNIE预训练模型常用序列标注模型，可以通过指定模型名字完成一键加载.PaddleNLP为了方便用户处理数据，内置了对于各个预训练模型对应的Tokenizer，可以完成文本token化，转token ID，文本长度截断等操作。**

本次选用的是RoBERTa large中文模型优化模型
from paddlenlp.transformers import RobertaForTokenClassification, RobertaTokenizer

model = RobertaForTokenClassification.from_pretrained(
    "roberta-wwm-ext-large",
    num_classes=(len(label_map) - 2) * 2 + 2)
tokenizer = RobertaTokenizer.from_pretrained("roberta-wwm-ext-large")    


文本数据处理直接调用tokenizer即可输出模型所需输入数据。

```python
inputs = tokenizer(text="请输入测试样例", max_seq_len=20)
# {'input_ids': [1, 647, 789, 109, 558, 525, 314, 656, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 'seq_len': 9}
```

### 2：加载处理数据

将数据集解压后更改名称train_data.json, dev_data.json, test_data.json将其放入relation_extraction/data中

我们可以加载自定义数据集。通过继承paddle.io.Dataset，自定义实现getitem 和 len两个方法。
如下代码已完成加载数据集操作：
from typing import Optional, List, Union, Dict

import numpy as np
import paddle
from tqdm import tqdm

from paddlenlp.transformers import ErnieTokenizer
from paddlenlp.utils.log import logger

from data_loader import parse_label, DataCollator, convert_example_to_feature
from extract_chinese_and_punct import ChineseAndPunctuationExtractor


class DuIEDataset(paddle.io.Dataset):
    def __init__(self, data, label_map, tokenizer, max_length=512, pad_to_max_length=False):
        super(DuIEDataset, self).__init__()

        self.data = data
        self.chn_punc_extractor = ChineseAndPunctuationExtractor()
        self.tokenizer = tokenizer
        self.max_seq_length = max_length
        self.pad_to_max_length = pad_to_max_length
        self.label_map = label_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        example = json.loads(self.data[item])
        input_feature = convert_example_to_feature(
            example, self.tokenizer, self.chn_punc_extractor,
            self.label_map, self.max_seq_length, self.pad_to_max_length)
        return {
            "input_ids": np.array(input_feature.input_ids, dtype="int64"),
            "seq_lens": np.array(input_feature.seq_len, dtype="int64"),
            "tok_to_orig_start_index":
            np.array(input_feature.tok_to_orig_start_index, dtype="int64"),
            "tok_to_orig_end_index": 
            np.array(input_feature.tok_to_orig_end_index, dtype="int64"),
            # If model inputs is generated in `collate_fn`, delete the data type casting.
            "labels": np.array(input_feature.labels, dtype="float32"),
        }


    @classmethod
    def from_file(cls,
                  file_path,
                  tokenizer,
                  max_length=512,
                  pad_to_max_length=None):
        assert os.path.exists(file_path) and os.path.isfile(
            file_path), f"{file_path} dose not exists or is not a file."
        label_map_path = os.path.join(
            os.path.dirname(file_path), "predicate2id.json")
        assert os.path.exists(label_map_path) and os.path.isfile(
            label_map_path
        ), f"{label_map_path} dose not exists or is not a file."
        with open(label_map_path, 'r', encoding='utf8') as fp:
            label_map = json.load(fp)

        with open(file_path, "r", encoding="utf-8") as fp:
            data = fp.readlines()
            return cls(data, label_map, tokenizer, max_length, pad_to_max_length)

#读取加载数据，设置batch——size大小
data_path = 'data'
batch_size = 8
max_seq_length = 128

train_file_path = os.path.join(data_path, 'train_data.json')
train_dataset = DuIEDataset.from_file(
    train_file_path, tokenizer, max_seq_length, True)
train_batch_sampler = paddle.io.BatchSampler(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
collator = DataCollator()
train_data_loader = paddle.io.DataLoader(
    dataset=train_dataset,
    batch_sampler=train_batch_sampler,
    collate_fn=collator)

eval_file_path = os.path.join(data_path, 'dev_data.json')
test_dataset = DuIEDataset.from_file(
    eval_file_path, tokenizer, max_seq_length, True)
test_batch_sampler = paddle.io.BatchSampler(
    test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_data_loader = paddle.io.DataLoader(
    dataset=test_dataset,
    batch_sampler=test_batch_sampler,
    collate_fn=collator)

#定义损失函数
import paddle.nn as nn

class BCELossForDuIE(nn.Layer):
    def __init__(self, ):
        super(BCELossForDuIE, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, labels, mask):
        loss = self.criterion(logits, labels)
        mask = paddle.cast(mask, 'float32')
        loss = loss * mask.unsqueeze(-1)
        loss = paddle.sum(loss.mean(axis=2), axis=1) / paddle.sum(mask, axis=1)
        loss = loss.mean()
        return loss

from utils import write_prediction_results, get_precision_recall_f1, decoding

@paddle.no_grad()
def evaluate(model, criterion, data_loader, file_path, mode):
    """
    mode eval:
    eval on development set and compute P/R/F1, called between training.
    mode predict:
    eval on development / test set, then write predictions to \
        predict_test.json and predict_test.json.zip \
        under /home/aistudio/relation_extraction/data dir for later submission or evaluation.
    """
    example_all = []
    with open(file_path, "r", encoding="utf-8") as fp:
        for line in fp:
            example_all.append(json.loads(line))
    id2spo_path = os.path.join(os.path.dirname(file_path), "id2spo.json")
    with open(id2spo_path, 'r', encoding='utf8') as fp:
        id2spo = json.load(fp)

    model.eval()
    loss_all = 0
    eval_steps = 0
    formatted_outputs = []
    current_idx = 0
    for batch in tqdm(data_loader, total=len(data_loader)):
        eval_steps += 1
        input_ids, seq_len, tok_to_orig_start_index, tok_to_orig_end_index, labels = batch
        logits = model(input_ids=input_ids)
        mask = (input_ids != 0).logical_and((input_ids != 1)).logical_and((input_ids != 2))
        loss = criterion(logits, labels, mask)
        loss_all += loss.numpy().item()
        probs = F.sigmoid(logits)
        logits_batch = probs.numpy()
        seq_len_batch = seq_len.numpy()
        tok_to_orig_start_index_batch = tok_to_orig_start_index.numpy()
        tok_to_orig_end_index_batch = tok_to_orig_end_index.numpy()
        formatted_outputs.extend(decoding(example_all[current_idx: current_idx+len(logits)],
                                          id2spo,
                                          logits_batch,
                                          seq_len_batch,
                                          tok_to_orig_start_index_batch,
                                          tok_to_orig_end_index_batch))
        current_idx = current_idx+len(logits)
    loss_avg = loss_all / eval_steps
    print("eval loss: %f" % (loss_avg))

    if mode == "predict":
        predict_file_path = os.path.join("/home/aistudio/relation_extraction/data", 'predictions.json')
    else:
        predict_file_path = os.path.join("/home/aistudio/relation_extraction/data", 'predict_eval.json')

    predict_zipfile_path = write_prediction_results(formatted_outputs,
                                                    predict_file_path)

    if mode == "eval":
        precision, recall, f1 = get_precision_recall_f1(file_path,
                                                        predict_zipfile_path)
        os.system('rm {} {}'.format(predict_file_path, predict_zipfile_path))
        return precision, recall, f1
    elif mode != "predict":
        raise Exception("wrong mode for eval func")

from paddlenlp.transformers import LinearDecayWithWarmup

learning_rate = 2e-5
num_train_epochs = 5
warmup_ratio = 0.06

criterion = BCELossForDuIE()
# Defines learning rate strategy.
steps_by_epoch = len(train_data_loader)
num_training_steps = steps_by_epoch * num_train_epochs
lr_scheduler = LinearDecayWithWarmup(learning_rate, num_training_steps, warmup_ratio)
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    apply_decay_param_fun=lambda x: x in [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])])



在训练过程中，模型保存在当前目录checkpoints文件夹下。同时在训练的同时使用官方评测脚本进行评估，输出P/R/F1指标。
在验证集上F1可以达到69.42。


### 4：开始训练

```shell
!bash train.sh

```

### 5.模型评估

```shell
!bash predict.sh

```

预测结果会被保存在data/predictions.json，data/predictions.json.zip，其格式与原数据集文件一致。
预测结果的名称也可能是duie.json.zip。
之后可以使用官方评估脚本评估训练模型在dev_data.json上的效果。如：
将得到预测结果移入relation_extraction中。
```shell
python re_official_evaluation.py --golden_file=dev_data.json  --predict_file=predicitons.json.zip [--alias_file alias_dict]
```

输出指标为Precision, Recall 和 F1，Alias file包含了合法的实体别名.


