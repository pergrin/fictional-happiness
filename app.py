#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('pip install Flask transformers nltk')
from flask import Flask, request, jsonify
import json, pprint, os, uuid, random, bisect, sys, torch, pickle, transformers, nltk
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import BertConfig, BertModel, BertPreTrainedModel, AdamW, BertTokenizer
from datetime import datetime
from torch.utils.data import Subset, RandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, f1_score
from tqdm import tqdm
from io import BytesIO
import nltk.data
nltk.download('punkt')
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# In[11]:


app = Flask(__name__)

pre_trained_model_type = 'bert-base-uncased'
model_path = 'bert-base-uncased'
model_save_path = './'
RECORD_PATH = ''
DATA_PATH = ''
train_frac = 1.0
results_path = model_save_path
weighted_multitask = True
learn_multitask = False
batch_size = 8
grad_acc_steps = 8
train_all = False
ignore_index = 0
weighted_category = False
task_learning_rate_fac = 100
grad_acc = True
oversample = False
epochs = 30
prop_drop = 0.2
entity_types = 7
relation_types = 7
entity_or_relations = 'relation'
train_further = False
checkpoint = epochs-1
train_dataset = 'Training'
dev_dataset = 'Test'
test_dataset = 'Test'
neg_entity_count = 150
neg_relation_count = 200
patience = 30
lr = 5e-5
lr_warmup = 0.1
weight_decay = 0.01
max_grad_norm = 1.0
width_embedding_size = 25
max_span_size = 10
max_pairs = 1000
relation_filter_threshold=0.3
is_overlapping = False
freeze_transformer = False

tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='bert-base-uncased')
UNK_TOKEN, CLS_TOKEN, SEP_TOKEN = 100, 101, 102


def generate_entity_mask(doc, is_training, neg_entity_count, max_span_size):
    sentence_length = doc['data_frame'].shape[0]
    entity_pool = set()
    for index_word in range(sentence_length):
        if index_word == 0 or doc['data_frame'].at[index_word, 'words']!=doc['data_frame'].at[index_word-1, 'words']:
            i=0
            for r in range(index_word+1, sentence_length+1):
                if r==sentence_length or doc['data_frame'].at[r, 'words']!=doc['data_frame'].at[r-1, 'words']:
                    entity_pool.add((index_word, r))
                    i+=1
                    if i>= max_span_size: break
    entity_mask, entity_label, entity_span = [], [], []
    for key in doc['entity_position']:
        index_word, r = doc['entity_position'][key]
        entity_pool.discard((index_word, r))
        entity_mask.append([0]*index_word+[1]*(r-index_word)+[0]*(sentence_length-r))
        entity_label.append(doc['data_frame'].at[index_word, 'entity_embedding'])
        entity_span.append((index_word, r, doc['data_frame'].at[index_word, 'entity_embedding']))
    if is_training:
        for index_word, r in random.sample(entity_pool, min(len(entity_pool), neg_entity_count)):
            entity_mask.append([0]*index_word + [1] * (r-index_word) + [0] * (sentence_length - r))
            entity_label.append(0)
    else:
        for index_word, r in entity_pool:
            entity_mask.append([0]*index_word + [1]*(r-index_word) + [0]*(sentence_length-r))
            entity_label.append(0)
    if len(entity_mask)>1 and is_training:
        tmp = list(zip(entity_mask, entity_label))
        random.shuffle(tmp)
        entity_mask, entity_label = zip(*tmp)
    return torch.tensor(entity_mask, dtype=torch.long), torch.tensor(entity_label, dtype=torch.long), entity_span

def generate_relation_mask(doc, is_training, neg_relation_count):	
    sentence_length = doc['data_frame'].shape[0]
    relation_pool = set([(e1, e2) for e1 in doc['entity_position'].keys() for e2 in doc['entity_position'].keys() if e1!=e2])
    relation_mask, relation_label, relation_span = [], [], []

    for key in doc['relations']:
        relation_pool.discard((doc['relations'][key]['source'], doc['relations'][key]['target']))
        relation_pool.discard((doc['relations'][key]['target'], doc['relations'][key]['source']))
        e1 = doc['entity_position'][doc['relations'][key]['source']]
        e2 = doc['entity_position'][doc['relations'][key]['target']]
        c = (min(e1[1], e2[1]), max(e1[0], e2[0]))
        template = [1] * sentence_length
        template[e1[0]:e1[1]] = [x*2 for x in template[e1[0]:e1[1]]]
        template[e2[0]:e2[1]] = [x*3 for x in template[e2[0]:e2[1]]]
        template[c[0]:c[1]] = [x*5 for x in template[c[0]:c[1]]]
        relation_mask.append(template)
        relation_label.append(doc['relations'][key]['type'])
        relation_span.append(((e1[0], e1[1], doc['data_frame'].at[e1[0], 'entity_embedding']),(e2[0], e2[1], doc['data_frame'].at[e2[0], 'entity_embedding']),				      doc['relations'][key]['type']))
    if is_training:
        for first, second in random.sample(relation_pool, min(len(relation_pool), neg_relation_count)):
            e1 = doc['entity_position'][first]
            e2 = doc['entity_position'][second]
            c = (min(e1[1], e2[1]), max(e1[0], e2[0]))
            template = [1] * sentence_length
            template[e1[0]:e1[1]] = [x*2 for x in template[e1[0]:e1[1]]]
            template[e2[0]:e2[1]] = [x*3 for x in template[e2[0]:e2[1]]]
            template[c[0]:c[1]] = [x*5 for x in template[c[0]:c[1]]]
            relation_mask.append(template)
            relation_label.append(0)
    if len(relation_mask)>1:
        tmp = list(zip(relation_mask, relation_label))
        random.shuffle(tmp)
        relation_mask, relation_label = zip(*tmp)
    return torch.tensor(relation_mask, dtype=torch.long), torch.tensor(relation_label, dtype=torch.long), relation_span


# In[12]:


def doc_to_input(doc, device, is_training=True, neg_entity_count=100, neg_relation_count=100, max_span_size=10):
    input_ids = [CLS_TOKEN] + doc['data_frame']['token_ids'].tolist() + [SEP_TOKEN]
    entity_mask, entity_label, entity_span = generate_entity_mask(doc, is_training, neg_entity_count, max_span_size)
    assert entity_mask.shape[1]==len(input_ids)-2
    relation_mask, relation_label, relation_span = generate_relation_mask(doc, is_training, neg_relation_count)
    if not torch.equal(relation_mask, torch.tensor([], dtype=torch.long)):
        assert relation_mask.shape[1] == len(input_ids)-2
    try:
        return {'input_ids': torch.tensor([input_ids]).long().to(device),
            'attention_mask': torch.ones((1, len(input_ids)), dtype=torch.long).to(device),
            'token_type_ids': torch.zeros((1, len(input_ids)), dtype=torch.long).to(device),
            'entity_mask': entity_mask.to(device),
            'entity_label': entity_label.to(device),
            'relation_mask': relation_mask.to(device),
            'relation_label': relation_label.to(device)},\
               {'document_name': doc['document_name'],
            'words': doc['data_frame']['words'],
            'entity_embedding': doc['data_frame']['entity_embedding'],
            'entity_span': entity_span,
            'relation_span': relation_span}
    except:
        return {'input_ids': torch.tensor([input_ids]).long().to(device),
            'attention_mask': torch.ones((1, len(input_ids)), dtype=torch.long).to(device),
            'token_type_ids': torch.zeros((1, len(input_ids)), dtype=torch.long).to(device),
            'entity_mask': entity_mask.to(device),
            'entity_label': entity_label.to(device),
            'relation_mask': relation_mask.to(device),
            'relation_label': relation_label.to(device)},\
               {'words': doc['data_frame']['words'],
            'entity_embedding': doc['data_frame']['entity_embedding'],
            'entity_span': entity_span,
            'relation_span': relation_span}


class Joint_Model(BertPreTrainedModel):
    def __init__(self, config: BertConfig, relation_types: int, entity_types: int, width_embedding_size: int, prop_drop: float, 
           max_pairs: int):
        super(Joint_Model, self).__init__(config)
        self.bert = BertModel(config)
        self.relation_classifier = nn.Linear(config.hidden_size*3 + width_embedding_size*2, relation_types)
        self.entity_classifier = nn.Linear(config.hidden_size*2 + width_embedding_size, entity_types)
        self.width_embedding = nn.Embedding(100, width_embedding_size)
        self.dropout = nn.Dropout(prop_drop)

        self._hidden_size = config.hidden_size
        self._relation_types = relation_types
        self._entity_types = entity_types
        self._relation_filter_threshold = relation_filter_threshold
        self._relation_possibility = relation_possibility
        self._max_pairs = max_pairs
        self._is_overlapping = is_overlapping
        self.init_weights()
        self.sigma = nn.Parameter(torch.zeros(2))

        if freeze_transformer:
            for param in self.bert.parameters(): param.requires_grad = False
      
    def _classify_entity(self, token_embedding, width_embedding, cls_embedding, entity_mask, entity_label, entity_weights):
        sentence_length = token_embedding.shape[0]
        hidden_size = token_embedding.shape[1]
        entity_count = entity_mask.shape[0]

        entity_embedding = token_embedding.view(1, sentence_length, hidden_size)+       ((entity_mask==0) * (-1e30)).view(entity_count, sentence_length, 1)
        entity_embedding = entity_embedding.max(dim=-2)[0]

        entity_embedding = torch.cat([cls_embedding.repeat(entity_count, 1), entity_embedding, width_embedding], dim=1)
        entity_embedding = self.dropout(entity_embedding)

        entity_logit = self.entity_classifier(entity_embedding)
        enitity_loss = None
        if entity_label is not None:
            loss_fct = CrossEntropyLoss(weight=entity_weights, reduction='none')
            entity_loss = loss_fct(entity_logit, entity_label)
            entity_loss = entity_loss.sum()/entity_loss.shape[-1]
        entity_confidence, entity_pred = F.softmax(entity_logit, dim=-1).max(dim=-1)
        return entity_logit, entity_loss, entity_confidence, entity_pred

    def _filter_span(self, entity_mask: torch.tensor, entity_pred: torch.tensor, entity_confidence: torch.tensor):
        entity_count = entity_mask.shape[0]
        sentence_length = entity_mask.shape[1]
        entities = [(entity_mask[i], entity_pred[i].item(), entity_confidence[i].item()) for i in range(entity_count)]
        entities = sorted(entities, key=lambda entity: entity[2], reverse=True)

        entity_span = []
        entity_embedding = torch.zeros((sentence_length,)) if not self._is_overlapping else None
        entity_type_map = {}
    
        for i in range(entity_count):
            e_mask, e_pred, e_confidence = entities[i]
            begin = torch.argmax(e_mask).item()
            end = sentence_length - torch.argmax(e_mask.flip(0)).item()

            assert end>begin
            assert e_mask[begin:end].sum() == end-begin
            assert e_mask.sum() == end-begin

            entity_type_map[(begin, end)] = e_pred

        if e_pred!=0:
            if self._is_overlapping: entity_span.append((begin, end, e_pred))
            elif not self._is_overlapping and entity_embedding[begin:end].sum() == 0:
                entity_span.append((begin, end, e_pred))
                entity_embedding[begin:end] = e_pred
        return entity_span, entity_embedding, entity_type_map

    def _generate_relation_mask(self, entity_span, sentence_length):
        relation_mask = []
        relation_possibility = []
        for e1 in entity_span:
            for e2 in entity_span:
                if e1!=e2:
                    c = (min(e1[1], e2[1]), max(e1[0], e2[0]))
                    template = [1]*sentence_length
                    template[e1[0]:e1[1]] = [x*2 for x in template[e1[0]:e1[1]]]
                    template[e2[0]:e2[1]] = [x*3 for x in template[e2[0]:e2[1]]]
                    template[c[0]:c[1]] = [x*5 for x in template[c[0]:c[1]]]
                    relation_mask.append(template)
                    if self._relation_possibility is not None:
                        if (e1[2], e2[2]) in self._relation_possibility:
                            relation_possibility.append(self._relation_possibility[(e1[2], e2[2])])
                        else: relation_mask.pop()
        return torch.tensor(relation_mask, dtype=torch.long).to(self.device), torch.tensor(relation_possibility, dtype=torch.long).to(self.device)

    def _classify_relation(self, token_embedding, e1_width_embedding, e2_width_embedding, relation_mask, relation_label, 
                         relation_possibility):
        sentence_length = token_embedding.shape[0]
        hidden_size = token_embedding.shape[1]
        relation_count = relation_mask.shape[0]

        e1_embedding = token_embedding.view(1, sentence_length, hidden_size)+     ((relation_mask%2!=0)*(-1e30)).view(relation_count, sentence_length, 1)
        e1_embedding = e1_embedding.max(dim=-2)[0]

        e2_embedding = token_embedding.view(1, sentence_length, hidden_size) +     ((relation_mask%3!=0)*(-1e30)).view(relation_count, sentence_length, 1)
        e2_embedding = e2_embedding.max(dim=-2)[0]

        c_embedding = token_embedding.view(1, sentence_length, hidden_size) +     ((relation_mask%5!=0)*(-1e30)).view(relation_count, sentence_length, 1)
        c_embedding = c_embedding.max(dim=-2)[0]
        c_embedding[c_embedding<-1e15] = 0

        relation_embedding = torch.cat([c_embedding, e1_embedding, e2_embedding, e1_width_embedding, e2_width_embedding], dim=1)
        relation_embedding = self.dropout(relation_embedding)

        relation_logit = self.relation_classifier(relation_embedding)
        relation_loss = None
        if relation_label is not None:
            loss_fct = BCEWithLogitsLoss(reduction='none')
            onehot_relation_label = F.one_hot(relation_label, num_classes=self._relation_types + 1).float()
            onehot_relation_label = onehot_relation_label[::, 1:]
            relation_loss = loss_fct(relation_logit, onehot_relation_label)
            relation_loss = relation_loss.sum(dim=-1)/relation_loss.shape[-1]
            relation_loss = relation_loss.sum()

        relation_sigmoid = torch.sigmoid(relation_logit)
        relation_sigmoid[relation_sigmoid < self._relation_filter_threshold] = 0
        relation_sigmoid = torch.cat([torch.zeros((relation_sigmoid.shape[0], 1)).to(self.device), relation_sigmoid], dim=-1)

        if self._relation_possibility is not None and relation_possibility is not None and     not torch.equal(relation_possibility, torch.tensor([], dtype=torch.long).to(self.device)):
            relation_sigmoid = torch.mul(relation_sigmoid, relation_possibility)
        relation_confidence, relation_pred = relation_sigmoid.max(dim=-1)

        return relation_logit, relation_loss, relation_confidence, relation_pred

    def _filter_relation(self, relation_mask: torch.tensor, relation_pred: torch.tensor, entity_type_map):
        relation_count = relation_mask.shape[0]
        sentence_length = relation_mask.shape[1]
        relation_span = []

        for i in range(relation_count):
            if relation_pred[i]!=0:
                e1_begin = torch.argmax((relation_mask[i]%2==0).long()).item()
                e1_end = sentence_length - torch.argmax((relation_mask[i].flip(0)%2==0).long()).item()
                assert e1_end>e1_begin
                assert (relation_mask[i, e1_begin:e1_end]%2).sum()==0

                e2_begin = torch.argmax((relation_mask[i]%3==0).long()).item()
                e2_end = sentence_length - torch.argmax((relation_mask[i].flip(0)%3==0).long()).item()
                assert e2_end>e2_begin
                assert (relation_mask[i, e2_begin:e2_end]%3).sum()==0

                relation_span.append(((e1_begin, e1_end, entity_type_map[(e1_begin, e1_end)]),
                          (e2_begin, e2_end, entity_type_map[(e2_begin, e2_end)]),
                          relation_pred[i].item()))
        return relation_span

    def forward(self, entity_weights, input_ids: torch.tensor, attention_mask: torch.tensor, token_type_ids: torch.tensor,
      entity_mask: torch.tensor = None, entity_label: torch.tensor = None,
      relation_mask: torch.tensor = None, relation_label: torch.tensor = None,
      is_training: bool = True):
        bert_embedding = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)['last_hidden_state']
        bert_embedding = torch.reshape(bert_embedding, (-1, self._hidden_size))
        cls_embedding = bert_embedding[:1]
        token_embedding = bert_embedding[1:-1]

        width_embedding = self.width_embedding(torch.sum(entity_mask, dim=-1))
        entity_logit, entity_loss, entity_confidence, entity_pred = self._classify_entity(token_embedding, width_embedding, cls_embedding, 
                                                                                          entity_mask, entity_label, 
                                                                                          entity_weights.to(self.device))

        entity_span, entity_embedding, entity_type_map = self._filter_span(entity_mask, entity_pred, entity_confidence)
        relation_possibility = None
        if not is_training or relation_mask is None:
            relation_mask, relation_possibility = self._generate_relation_mask(entity_span, token_embedding.shape[0])
            relation_label = None

        output = {'loss': entity_loss,'entity': {'logit': entity_logit,'loss': None if entity_loss is None else entity_loss.item(),
                                             'pred': entity_pred,'confidence': entity_confidence,'span': entity_span,
                                             'embedding': entity_embedding},'relation': None}

        if relation_mask is None or torch.equal(relation_mask, torch.tensor([], dtype=torch.long).to(self.device)): return output

        relation_count = relation_mask.shape[0]
        relation_logit = torch.zeros((relation_count, self._relation_types))
        relation_loss = []
        relation_confidence = torch.zeros((relation_count,))
        relation_pred = torch.zeros((relation_count,), dtype=torch.long)
        e1_width_embedding = self.width_embedding(torch.sum(relation_mask%2==0, dim=-1))
        e2_width_embedding = self.width_embedding(torch.sum(relation_mask%3==0, dim=-1))
        for i in range(0, relation_count, self._max_pairs):
            j = min(relation_count, i+self._max_pairs)
            logit, loss, confidence, pred = self._classify_relation(token_embedding,
                        e1_width_embedding[i:j],
                        e2_width_embedding[i:j],
                        relation_mask[i:j],
                        relation_label[i:j] if relation_label is not None else None,
                        relation_possibility[i:j] if relation_possibility is not None else None)

            relation_logit[i:j] = logit
            if loss is not None: relation_loss.append(loss)
            relation_confidence[i:j] = confidence
            relation_pred[i:j] = pred
    
        relation_loss = None if len(relation_loss)==0 else (sum(relation_loss)/relation_count)
        relation_span = self._filter_relation(relation_mask, relation_pred, entity_type_map)
    
        if relation_loss is not None:
            output['loss'] = 1.3*relation_loss + 0.7*entity_loss
        output['relation'] = {
            'logit': relation_logit,
            'loss': None if relation_loss is None else relation_loss.item(),
            'pred': relation_pred,
            'confidence': relation_confidence,
            'span': relation_span}
        return output

entity_encode = {'None':0, 'Generic':1, 'Material':2, 'Method':3, 'Metric':4, 'OtherScientificTerm':5, 'Task':6}
relation_encode = {'None':0, 'COMPARE':1, 'CONJUNCTION':2, 'EVALUATE-FOR':3, 'FEATURE-OF':4, 'HYPONYM-OF':5, 'PART-OF': 6, 'USED-FOR':7}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
entity_label_map = {v:k for k, v in entity_encode.items()}
entity_classes = list(entity_label_map.keys())
entity_classes.remove(0)

relation_label_map = {v:k for k, v in relation_encode.items()}
relation_classes = list(relation_label_map.keys())
relation_classes.remove(0)

tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='bert-base-uncased')

relation_possibility=None



# In[7]:


config = BertConfig.from_pretrained('bert-base-uncased')
neural_model = Joint_Model.from_pretrained('bert-base-uncased', config=config, relation_types = relation_types,
            entity_types = entity_types, width_embedding_size = width_embedding_size,
            prop_drop = prop_drop, max_pairs=max_pairs)
neural_model.to(device)


# In[13]:


state_dict = torch.load('./model/joint_model.model', map_location=device)
neural_model.load_state_dict(state_dict, strict=False)


# In[33]:


def predict(base_text):
    with app.app_context():
        try:
            sentences = sent_tokenizer.tokenize(base_text)
            results = {}
            fin_result = {}
            entity_weights = torch.tensor([1.0]*len(range(entity_types)))
            for sent_num, sentence in enumerate(sentences):
                word_list = sentence.split()
                words, token_ids = [], []
                for word in word_list:
                    token_id = tokenizer(word)["input_ids"][1:-1]
                    for tid in token_id:
                        words.append(word)
                        token_ids.append(tid)
                data_frame = pd.DataFrame()
                data_frame['words'] = words
                data_frame['token_ids'] = token_ids
                data_frame['entity_embedding'] = 0
                data_frame['sentence_embedding'] = 0
                doc = {'data_frame': data_frame, 'entity_position':{}, 'entities':{}, 'relations':{}}
                inputs, infos = doc_to_input(doc, device, is_training=False, max_span_size = max_span_size)
                outputs = neural_model(entity_weights, **inputs, is_training=False)
                pred_entity_span = outputs['entity']['span']
                pred_relation_span = [] if outputs['relation'] is None else outputs['relation']['span']
                tokens = tokenizer.convert_ids_to_tokens(token_ids)
                dic = {'Entities':'', 'Relations':''}
                for begin, end, entity_type in pred_entity_span:
                    dic['Entities']+=entity_label_map[entity_type]+'|'+' '.join(tokens[begin:end])+'\n'
                for e1, e2, relation_type in pred_relation_span:
                    dic['Relations']+=relation_label_map[relation_type]+'|'+ ' '.join(tokens[e1[0]:e1[1]])+'...'+' '.join(tokens[e2[0]:e2[1]])+'\n'
                results['sent_'+str(sent_num)]=dic
            #print(type(str(results)))
            fin_result['prediction']=str(results)
            #print(fin_result)
            return jsonify(fin_result)
        except:
            print('Error occur in script generating!', e)
            return jsonify({'error': e}), 500

@app.route("/predict", methods=["POST"])
def main():
    try:
        base_text = request.form.get('base_text')

    except Exception as e:
        return jsonify({'message': 'Invalid request'}), 500

    prediction = predict(base_text)
    return prediction


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5000")

