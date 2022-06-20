import logging
import numpy as np
import pickle as pkl

import torch

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm, trange

from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.modeling import BertConfig, BertModel, BertPreTrainedModel
from pytorch_pretrained_bert.modeling import CONFIG_NAME, WEIGHTS_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.nn import BCEWithLogitsLoss

logger = logging.getLogger(__name__)

class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size, num_labels]
            with indices selected in [0, ..., num_labels].
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    """
    def __init__(self, config, num_labels=2, loss_fct="bbce"):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.loss_fct = loss_fct
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if labels is not None:
            if self.loss_fct == "bbce":
                loss_fct = BalancedBCEWithLogitsLoss()
            else:
                loss_fct = torch.nn.MultiLabelSoftMarginLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss
        else:
            return logits
    
    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

class InputFeatures(object):
    """A single set of features of data."""
    
    def __init__(self, input_ids, input_mask, segment_ids, label_ids, guid):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.guid = guid

class InputExample(object):
    """A single training/test example for multi-label classification."""
    
    def __init__(self, guid, text_a, text_b=None, labels=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            labels: (Optional) list of string. The labels of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels
    
def convert_examples_to_features(examples, label_list, tokenizer):

    """Loads a data file into a list of `InputBatch`s."""
    max_seq_length = 256
    label_map = {label : i for i, label in enumerate(label_list)} #see above
    
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        
        tokens_a = tokenizer.tokenize(example.text_a)
        
        tokens_b = None
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
        
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        
        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        
        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        label_ids = example.labels[:]
        
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %r" % label_ids)
        
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_ids=label_ids,
                              guid=example.guid))
    return features

def sigmoid(x):
        return 1. / (1. + np.exp(-x))

def predict(examples):
    test_examples = examples
    # test_examples = processor.get_test_examples()
    test_features = convert_examples_to_features(
        test_examples, label_list, tokenizer)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(test_examples))
    logger.info("  Batch size = %d", 1)
    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    all_doc_ids = torch.tensor([f.guid for f in test_features], dtype=torch.long)
    
    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_doc_ids)
    # Run prediction for full data
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=1)
    
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    ids = []
    # FIXME: make it flexible to accept path
    #all_ids_test = read_ids(os.path.join(args.data_dir, "ids_test.txt"))
    
    for input_ids, input_mask, segment_ids, doc_ids in tqdm(test_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        doc_ids = doc_ids.to(device)
        
        with torch.no_grad():
            logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=None)
        
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)
        if len(ids) == 0:
            ids.append(doc_ids.detach().cpu().numpy())
        else:
            ids[0] = np.append(
                ids[0], doc_ids.detach().cpu().numpy(), axis=0)
    
    ids = ids[0]
    preds = sigmoid(preds[0])
    preds = (preds > 0.5).astype(int)
    id2preds = {val:preds[i] for i, val in enumerate(ids)}
    
    with open(("./mlb.pkl"), "rb") as rf:
        mlb = pkl.load(rf)

    preds = [mlb.classes_[preds[i, :].astype(bool)].tolist() for i in range(preds.shape[0])]
    id2preds = {val:preds[i] for i, val in enumerate(ids)}
    #preds = [id2preds[val] if val in id2preds else [] for i, val in enumerate(all_ids_test)]
    
    return(id2preds)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load a trained model and vocabulary that you have fine-tuned

model = BertForMultiLabelSequenceClassification.from_pretrained("./model", num_labels=108)
tokenizer = BertTokenizer.from_pretrained("./model", do_lower_case=True)
model.to(device)


import pickle

with open('mlb.pkl', 'rb') as f:
    
    labels = pickle.load(f)
    
label_list = labels.classes_

from flask import Flask, redirect, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def main():
        if request.method == "GET":
            return render_template("main.html")

        else:
            # run the model on the input text and return the results
            text = request.form.get("text")
            print(text)
            examples = []

            text_a = "\n".join(text.replace("<SENT>", "").split("<SECTION>"))
            text_b = None
            labels = []
            guid = 1

            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,labels=labels))

            results = predict(examples)

            with open('codes_and_titles_en.txt') as f:
                lines = f.readlines()
            
            codes = {}
            for line in lines:
                code = (line.strip("\n").split("\t"))
                codes[code[1]]=code[0]
            
            return render_template("main.html", text = text, results=results, codes = codes)


