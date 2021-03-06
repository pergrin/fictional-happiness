import ast
import json, pprint, os, uuid, random, bisect, sys, torch, pickle, transformers
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

inp_data = json.load(open('sciERC_raw.json', r))
formatted_data = []
all_rels, all_mentions = [], []
poss_rels = {}

entity_encode = {'None':0, 'Generic':1, 'Material':2, 'Method':3, 'Metric':4, 'OtherScientificTerm':5, 'Task':6}
relation_encode = {'None':0, 'COMPARE':1, 'CONJUNCTION':2, 'EVALUATE-FOR':3, 'FEATURE-OF':4, 'HYPONYM-OF':5, 'PART-OF': 6, 'USED-FOR':7}
relation_possibility={}

def make_phrases(toks):
  sent_str=''
  for tok_num in range(len(toks)):
    if sent_str!='' and sent_str[-1]=='-':
      sent_str+=toks[tok_num]
    elif toks[tok_num][0].isalnum():
      sent_str+=' '+toks[tok_num]
    else:
      sent_str+=toks[tok_num]
  return sent_str

for dic in inp_data:	
  annotations={}
  sents=[]
  for ele in dic['sentences']:
    sents.append(make_phrases(ele))
  if sents[0][0]==' ': sents[0]=sents[0][1:]
  full_text = ''.join(sents)
  annotations['id']=str(uuid.uuid4())
  annotations['text']=full_text
  annotations['sentences']=[]
  annotations['mentions']=[]
  annotations['relations']=[]
  for i, sent in enumerate(dic['sentences']):
    sent_dic={}
    sent_dic['id']='s'+str(i)
    sent_dic['text']=sents[i]
    sent_dic['begin']=full_text.find(sent_dic['text'])
    sent_dic['end']=sent_dic['begin']+len(sent_dic['text'])
    sent_dic['tokens']=[]			
    prev_tok_end = 0
    for j, tok in enumerate(sent):
      tok_dic={}
      tok_dic['id']=sent_dic['id']+'-t'+str(j)
      tok_dic['text']=tok
      tok_dic['begin']=sent_dic['begin']+sent_dic['text'].find(tok, prev_tok_end)
      tok_dic['end']=tok_dic['begin']+len(tok_dic['text'])
      prev_tok_end=tok_dic['end']-sent_dic['begin']
      sent_dic['tokens'].append(tok_dic)
    annotations['sentences'].append(sent_dic)
  ner = []
  for i in dic['ner']:
    for j in i:
      ner.append(j)
  relations = []
  for i in dic['relations']:
    for j in i:
      relations.append(j)
  ment_map, ignore_ments={}, []
  all_tokens = []
  for ele in dic['sentences']:
    all_tokens+=ele
  prev_ment_end = 0
  ment_count = 0
  for i, ment in enumerate(ner):
    ment_dic={}
    ment_text = make_phrases(all_tokens[ment[0]:ment[1]+1])[1:]
    ment_dic['begin'] = annotations['text'].find(ment_text, prev_ment_end)
    ment_dic['end'] = ment_dic['begin'] + len(ment_text)
    prev_ment_end = ment_dic['end']
    ment_dic['type'] = ment[2]
    ment_dic['text'] = ment_text
    all_mentions.append(ment_dic['type'])
    if ment_dic['begin']!=-1: 
      ment_dic['id']='m'+str(ment_count)
      ment_count+=1
      ment_map[str(ment[0])+','+str(ment[1])]=[ment_dic['id'], ment_dic['type']]
      annotations['mentions'].append(ment_dic)
    else: ignore_ments.append(str(ment[0])+','+str(ment[1]))

  for i, rel in enumerate(relations):
    if str(rel[0])+','+str(rel[1]) in ignore_ments or str(rel[2])+','+str(rel[3]) in ignore_ments: continue
    rel_dic={}
    rel_dic['id']='r'+str(i)
    rel_dic['type']=rel[-1]
    rel_dic['args']=[ment_map[str(rel[0])+','+str(rel[1])][0], ment_map[str(rel[2])+','+str(rel[3])][0]]
    poss_rels[ment_map[str(rel[0])+','+str(rel[1])][1]+','+ment_map[str(rel[2])+','+str(rel[3])][1]]=rel_dic['type']
    all_rels.append(rel_dic['type'])
    annotations['relations'].append(rel_dic)
  
  formatted_data.append(annotations)


random.shuffle(formatted_data)
train_files = formatted_data[:int(0.8*len(formatted_data))]
test_files = formatted_data[int(0.8*len(formatted_data)):]

print(np.unique(all_rels), np.unique(all_mentions))
pprint.pprint(poss_rels)

for k, v in poss_rels.items():
	left_ent = entity_encode[k.split(',')[0]]
	right_ent = entity_encode[k.split(',')[1]]
	eles = [0]*len(relation_encode)
	eles[relation_encode[v]]=1
	relation_possibility[(left_ent, right_ent)] = eles
pprint.pprint(relation_possibility)

"""# New Section"""

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

def get_word_doc(doc):
	words = []
	begins = []
	ends = []
	sentence_embedding = []
	sentence_count = 0
	for sentence in doc['sentences']:
		for word in sentence['tokens']:
			words.append(word['text'])
			begins.append(word['begin'])
			ends.append(word['end'])
			sentence_embedding.append(sentence_count)
		sentence_count+=1
	return words, begins, ends, sentence_embedding

def get_token_id(words):
	token_id = []
	for word in words:
		token_id.append(tokenizer(word)['input_ids'][1:-1])
	return token_id

def expand_token_id(token_id, words, begins, ends, sentence_embedding):
	assert len(token_id)==len(words)==len(begins)==len(ends)==len(sentence_embedding), 'input lists do not have the same length, abort'
	
	new_token_id = []
	new_words = []
	new_begins = []
	new_ends = []
	new_sentence_embedding = []
	for i in range(len(token_id)):
		for tid in token_id[i]:
			new_token_id.append(tid)
			new_words.append(words[i])
			new_begins.append(begins[i])
			new_ends.append(ends[i])
			new_sentence_embedding.append(sentence_embedding[i])
	return new_token_id, new_words, new_begins, new_ends, new_sentence_embedding

def get_entity_doc(doc, begins):
	entity_embedding = [0]*len(begins)
	entity_position = {}
	for mention in doc['mentions']:
		low = bisect.bisect_left(begins, mention['begin'])
		high = bisect.bisect_left(begins, mention['end'])
		entity_position[mention['id']] = (low, high)
		for i in range(low, high):		
			entity_embedding[i] = entity_encode[mention['type']]
	return entity_position, entity_embedding

def get_relation_doc(doc):
	relations = {}
	for relation in doc['relations']:
		relations[relation['id']] = {'type': relation_encode[relation['type']], 'source': relation['args'][0],\
					     'target': relation['args'][1]}
	return relations

def extract_doc(document):
	data_frame = pd.DataFrame()
	words, begins, ends, sentence_embedding = get_word_doc(document)
	token_ids = get_token_id(words)
	data_frame['token_ids'], data_frame['words'], data_frame['begins'], data_frame['ends'],\
	data_frame['sentence_embedding'] = expand_token_id(token_ids, words, begins, ends, sentence_embedding)
	data_frame['tokens'] = tokenizer.convert_ids_to_tokens(data_frame['token_ids'])
	entity_position, data_frame['entity_embedding'] = get_entity_doc(document, list(data_frame['begins']))
	relations = get_relation_doc(document)
	return {'document': '',\
		'data_frame': data_frame,\
		'entity_position': entity_position,\
		'relations': relations}

def extract_data(group):
	if group=='Train': docs=train_files
	if group=='Test': docs=test_files
	data=[]
	for document in docs: data.append(extract_doc(document))
	return data

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
		relation_span.append(((e1[0], e1[1], doc['data_frame'].at[e1[0], 'entity_embedding']),\
				      (e2[0], e2[1], doc['data_frame'].at[e2[0], 'entity_embedding']),\
				      doc['relations'][key]['type']))
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

def doc_to_input(doc, device, is_training=True, neg_entity_count=100, neg_relation_count=100, max_span_size=10):
	input_ids = [CLS_TOKEN] + doc['data_frame']['token_ids'].tolist() + [SEP_TOKEN]
	entity_mask, entity_label, entity_span = generate_entity_mask(doc, is_training, neg_entity_count, max_span_size)
	assert entity_mask.shape[1]==len(input_ids)-2
	relation_mask, relation_label, relation_span = generate_relation_mask(doc, is_training, neg_relation_count)
	if not torch.equal(relation_mask, torch.tensor([], dtype=torch.long)):
		assert relation_mask.shape[1] == len(input_ids)-2
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

def data_generator(group, device, is_training=True, neg_entity_count=100, neg_relation_count=100, max_span_size=10):
	data=extract_data(group)
	for document_number, doc in enumerate(data):
		sentence_id, starting_index = 0, 0
		doc['data_frame'].loc[doc['data_frame'].index.max() + 1, 'sentence_embedding']\
		= doc['data_frame']['sentence_embedding'].max() + 1
		for index, row in doc['data_frame'].iterrows():
			if row['sentence_embedding']!=sentence_id:
				if index-starting_index>510:
					starting_index = index-510
				tmp_entity_position = {}
				for entity in doc['entity_position']:
					if starting_index<=doc['entity_position'][entity][0] < doc['entity_position'][entity][1]<=index:
						tmp_entity_position[entity] = (
							doc['entity_position'][entity][0] - starting_index,
							doc['entity_position'][entity][1] - starting_index
						)
				tmp_relations={}
				for relations in doc['relations']:
 					if doc['relations'][relations]['source'] in tmp_entity_position and doc['relations'][relations]['target'] in tmp_entity_position:
						 tmp_relations[relations] = doc['relations'][relations]
				tmp_doc = {
					'document_name': str(document_number),\
					'data_frame': doc['data_frame'][starting_index:index].reset_index(drop=True),\
					'entity_position': tmp_entity_position,\
					'relations': tmp_relations}
				yield doc_to_input(tmp_doc, device, is_training, neg_entity_count, neg_relation_count, max_span_size)
				sentence_id = row['sentence_embedding']
				starting_index = index

def evaluate_results(true_labels, predicted_labels, label_map, classes):
	precision, recall, fbeta_score, support = \
		precision_recall_fscore_support(true_labels, predicted_labels, average=None, labels=classes, zero_division=0)
	result = pd.DataFrame(index=[label_map[c] for c in classes])
	result['precision'] = precision
	result['recall'] = recall
	result['fbeta_score'] = fbeta_score
	result['support'] = support
	result.loc['macro'] = list(precision_recall_fscore_support(true_labels, predicted_labels, average='macro', labels=classes, 
	                                                           zero_division=0))
	return result

def evaluate_f1_global(true_labels, pred_labels):
	return f1_score(true_labels, pred_labels, zero_division=0)

def evaluate_span(true_span, pred_span, label_map, classes):
	assert len(true_span)==len(pred_span)
	true_label, pred_label=[],[]
	for true_span_batch, pred_span_batch in zip(true_label, pred_span):
		true_span_batch = dict([((item[0][:2] if isinstance(item[0], tuple) else item[0],
					  item[1][:2] if isinstance(item[1], tuple) else item[1]),
					  item[2]) for item in true_span_batch])

		pred_span_batch = dict([((item[0][:2] if isinstance(item[0], tuple) else item[0],
					  item[1][:2] if isinstance(item[1], tuple) else item[1]),
					  item[2]) for item in pred_span_batch])

		s= set()
		s.update(true_span_batch.keys())
		s.update(pred_span_batch.keys())
		for span in s:
			if span in true_span_batch:
				true_label.append(true_span_batch[span])
			else: true_label.append(0)
			if span in pred_span_batch:
				pred_label.append(pred_span_batch[span])
			else: pred_label.append(0)
	
	assert len(true_label)==len(pred_label)
	return evaluate_results(true_label, pred_label, label_map, classes)

class Joint_Model(BertPreTrainedModel):
  def __init__(self, config: BertConfig, relation_types: int, entity_types: int, width_embedding_size: int, prop_drop: float, 
               max_pairs: int):
    super(Joint_Model, self).__init__(config)
    print(f'Config is (config)')
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

    entity_embedding = token_embedding.view(1, sentence_length, hidden_size)+ \
      ((entity_mask==0) * (-1e30)).view(entity_count, sentence_length, 1)
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

    e1_embedding = token_embedding.view(1, sentence_length, hidden_size)+ \
    ((relation_mask%2!=0)*(-1e30)).view(relation_count, sentence_length, 1)
    e1_embedding = e1_embedding.max(dim=-2)[0]
    
    e2_embedding = token_embedding.view(1, sentence_length, hidden_size) +\
     ((relation_mask%3!=0)*(-1e30)).view(relation_count, sentence_length, 1)
    e2_embedding = e2_embedding.max(dim=-2)[0]

    c_embedding = token_embedding.view(1, sentence_length, hidden_size) + \
    ((relation_mask%5!=0)*(-1e30)).view(relation_count, sentence_length, 1)
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

    if self._relation_possibility is not None and relation_possibility is not None and \
    not torch.equal(relation_possibility, torch.tensor([], dtype=torch.long).to(self.device)):
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
      if loss is not None:
        relation_loss.append(loss)
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

os.makedirs(model_save_path, exist_ok=True)
EPOCH_="epoch:"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
entity_label_map = {v:k for k, v in entity_encode.items()}
entity_classes = list(entity_label_map.keys())
entity_classes.remove(0)

relation_label_map = {v:k for k, v in relation_encode.items()}
relation_classes = list(relation_label_map.keys())
relation_classes.remove(0)

tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='bert-base-uncased')


def get_optimizer_params(model):
	param_optimizer = list(model.named_parameters())
	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.Weight']
	optimizer_params1 = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and n.startswith('bert')],
                       'weight_decay': weight_decay},
                       {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and n.startswith('bert')],
                        'weight_decay':0.0}]
	optimizer_params2 = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and not n.startswith('bert')],
                       'weight_decay': weight_decay},
                      {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and not n.startswith('bert')],
                       'weight_decay': 0.0}]
	return optimizer_params1, optimizer_params2


def take_first_tokens(embedding, words):
	reduced_embedding = []
	for i, word in enumerate(words):
		if i==0 or word!=words[i-1]:
			reduced_embedding.append(embedding[i])
	return reduced_embedding

train_generator = data_generator('Train', device, is_training=True, 
              neg_entity_count = neg_entity_count,
              neg_relation_count = neg_relation_count,
              max_span_size = max_span_size)
train_dataset = list(train_generator)
test_generator = data_generator('Test', device, is_training=True, 
              neg_entity_count = neg_entity_count,
              neg_relation_count = neg_relation_count,
              max_span_size = max_span_size)
test_dataset = list(test_generator)
config = BertConfig.from_pretrained('bert-base-uncased')
random.shuffle(train_dataset)
new_dataset = {'train': None, 'test': None, 'val': None}
new_dataset['train'], new_dataset['test'], new_dataset['val'] = train_dataset, test_dataset, test_dataset
entity_weights = torch.tensor([1.0]*len(range(entity_types)))
train_size = len(new_dataset['train'])
val_dataset = new_dataset['val']

def evaluate_val(neural_model, eval_dataset, epoch, data_val, entity_weights):
	neural_model.eval()
	eval_size = len(eval_dataset)
	eval_entity_span_pred = []
	eval_entity_span_true = []
	eval_entity_embedding_pred = []
	eval_entity_embedding_true = []
	eval_relation_span_pred = []
	eval_relation_span_true = []
	for inputs, infos in tqdm(eval_dataset, total=eval_size, desc='Evaluating the validation set'):
		outputs = neural_model(entity_weights, **inputs, is_training=False)
		eval_entity_span_pred.append(outputs['entity']['span'])
		eval_entity_span_true.append(infos['entity_span'])
	if not is_overlapping:
		eval_entity_embedding_pred+= take_first_tokens(outputs['entity']['embedding'].tolist(), infos['words'])
		eval_entity_embedding_true+= take_first_tokens(infos['entity_embedding'].tolist(), infos['words'])
		assert len(eval_entity_embedding_pred) == len(eval_entity_embedding_true)
	eval_relation_span_pred.append([] if outputs['relation'] is None else outputs['relation']['span'])
	eval_relation_span_true.append(infos['relation_span'])
	results = pd.concat([
		evaluate_span(eval_entity_span_true, eval_entity_span_pred, entity_label_map, entity_classes),
		evaluate_span(eval_relation_span_true, eval_relation_span_pred, relation_label_map, relation_classes),
		], keys = ['Entity span', 'Strict relation'])
	return results

neural_model = Joint_Model.from_pretrained('bert-base-uncased', config=config, relation_types = relation_types,
            entity_types = entity_types, width_embedding_size = width_embedding_size,
            prop_drop = prop_drop, max_pairs=max_pairs)

neural_model.to(device)
optimizer_params1, optimizer_params2 = get_optimizer_params(neural_model)
optimizer1 = AdamW(optimizer_params1, lr=lr, weight_decay=weight_decay, correct_bias=False)
scheduler1 = transformers.get_linear_schedule_with_warmup(optimizer1, num_warmup_steps=lr_warmup*train_size//batch_size*epochs, num_training_steps=train_size//batch_size*epochs)
optimizer2 = AdamW(optimizer_params2, lr=lr*task_learning_rate_fac, weight_decay=weight_decay,
        correct_bias=False)
scheduler2 = transformers.get_linear_schedule_with_warmup(optimizer2, 
                                                          num_warmup_steps=lr_warmup*train_size//batch_size\
                                                          *epochs, num_training_steps=train_size//batch_size*\
                                                          epochs)

for epoch in range(epochs):
  losses=[]
  entity_losses = []
  relation_losses = []
  train_entity_pred = []
  train_entity_true = []
  train_relation_pred = []
  train_relation_true = []
  neural_model.zero_grad()
  iter_count = 1
  
  for inputs, infos in tqdm(new_dataset['train'], total=train_size, desc='Train epoch %s'%epoch):
    neural_model.train()
    outputs=neural_model(entity_weights, **inputs, is_training=True)
    loss = outputs['loss']
    loss = loss/grad_acc_steps
    loss.backward()
    if iter_count%grad_acc_steps==0:
      torch.nn.utils.clip_grad_norm_(neural_model.parameters(), max_grad_norm)
      optimizer1.step()
      optimizer2.step()
      neural_model.zero_grad()
    if iter_count%batch_size==0:
      scheduler1.step()
      scheduler2.step()
    iter_count+=1
    losses.append(loss.item())
    entity_losses.append(outputs['entity']['loss'])
    if outputs['relation'] is not None:
      relation_losses.append(outputs['relation']['loss'])
    train_entity_pred+= outputs['entity']['pred'].tolist()
    train_entity_true+= inputs['entity_label'].tolist()
    train_relation_pred+= [] if outputs['relation'] is None else outputs['relation']['pred'].tolist()
    train_relation_true+= inputs['relation_label'].tolist()
    assert len(train_entity_pred) == len(train_entity_true)
    assert len(train_relation_pred) == len(train_relation_true)
  print(EPOCH_, epoch, 'average_loss:', sum(losses)/len(losses))
  print(EPOCH_, epoch, 'average entity loss:', sum(entity_losses)/len(entity_losses))
  print(EPOCH_, epoch, 'average relation loss:', sum(relation_losses)/len(relation_losses))
  results = pd.concat([
      evaluate_results(train_entity_true, train_entity_pred, entity_label_map, entity_classes),
      evaluate_results(train_relation_true, train_relation_pred, relation_label_map, relation_classes)]\
      , keys=['Entity', 'Relation'])

output_dicts = []
def convert(doc, full_text, out_folder):
	dic={}
	dic['model_info']='Joint model'
	dic['title']=doc[0]
	dic['paragraph']=full_text
	file_name=out_folder+doc[0]+'.json'
	dic['predicted_data']={'entities':[], 'relations':[]}
	ent_map, ent_count = {}, 1
	for sent in doc[1:]:
		sent_text=sent[0]
		for ent in sent[2]:
			if ent==[]:continue
			ent_dic={}
			ent_dic['category']=ent[-2]
			ent_dic['sentence']=sent_text
			ent_dic['startIndex']=ent[0]
			ent_dic['endIndex']=ent[1]
			ent_dic['text']=dic['paragraph'][ent_dic['startIndex']:ent_dic['endIndex']]
			ent_dic['id']=str(uuid.uuid4())
			ent_count+=1
			ent_map[str(ent[0])+ ' '+str(ent[1])]=ent_dic['id']
			dic['predicted_data']['entities'].append(ent_dic)

	for sent in doc[1:]:
		sent_text = sent[0]
		for rel in sent[-1]:
			if rel==[]: continue
			rel_dic={}
			rel_dic['inEntity']={}
			rel_dic['inEntity']['category']=rel[-2]
			rel_dic['inEntity']['startIndex']=rel[0]
			rel_dic['inEntity']['endIndex']=rel[1]
			rel_dic['inEntity']['text']=dic['paragraph'][rel_dic['inEntity']['startIndex']:rel_dic['inEntity']['endIndex']]
			if str(rel[0]) + ' ' + str(rel[1]) in ent_map.keys():
				rel_dic['inEntity']['id']=ent_map[str(rel[0])+' '+str(rel[1])]
			else:
				ent_dic={}
				ent_dic['category']=rel[-2]
				ent_dic['sentence']=sent_text
				ent_dic['startIndex']=rel[0]
				ent_dic['endIndex']=rel[1]
				ent_dic['text']=dic['paragraph'][ent_dic['startIndex']:ent_dic['endIndex']]
				ent_dic['id']=str(uuid.uuid4())
				ent_count+=1
				ent_map[str(rel[0]) + ' ' + str(rel[1])] = ent_dic['id']
				rel_dic['inEntity']['id'] =ent_dic['id']
				dic['predicted_data']['entities'].append(ent_dic)
			rel_dic['inEntity']['sentence']=sent_text
		
			rel_dic['outEntity']={}
			rel_dic['outEntity']['category']=rel[-1]
			rel_dic['outEntity']['startIndex']=rel[2]
			rel_dic['outEntity']['endIndex']=rel[3]
			rel_dic['outEntity']['text']=dic['paragraph'][rel_dic['outEntity']['startIndex']:rel_dic['outEntity']['endIndex']]
			if str(rel[2])+' '+str(rel[3]) in ent_map.keys():
				rel_dic['outEntity']['id']=ent_map[str(rel[2])+' '+str(rel[3])]
			else:
				ent_dic={}
				ent_dic['category']=rel[-1]
				ent_dic['sentence']=sent_text
				ent_dic['startIndex']=rel[2]
				ent_dic['endIndex']=rel[3]
				ent_dic['text']=dic['paragraph'][ent_dic['startIndex']:ent_dic['endIndex']]
				ent_dic['id']=str(uuid.uuid4())
				ent_count+=1
				ent_map[str(rel[2]) + ' ' + str(rel[3])] = ent_dic['id']
				rel_dic['outEntity']['id'] = ent_dic['id']
				dic['predicted_data']['entities'].append(ent_dic)
			rel_dic['outEntity']['sentence']=sent_text

			rel_dic['sentence']=sent_text
			rel_dic['category']=rel[4]
			rel_dic['id_pair']=[rel_dic['inEntity']['id'], rel_dic['outEntity']['id']]
			dic['predicted_data']['relations'].append(rel_dic)
	output_dicts.append(dic)

relation_possibility = None

def take_first_tokens(embedding, words):
	reduced_embedding = []
	for i, word in enumerate(words):
		if i==0 or word!=words[i-1]: reduced_embedding.append(embedding[i])
	return reduced_embedding



def get_results(eval_dataset, neural_model, category_weights):
	neural_model.eval()
	eval_size = len(eval_dataset)
	eval_entity_span_pred = []
	eval_entity_span_true = []
	eval_entity_embedding_pred = []
	eval_entity_embedding_true = []
	eval_relation_span_pred = []
	eval_relation_span_true = []
	prev_doc = ''
	doc = 0
	ent_rels = []

	for inputs, infos in tqdm(eval_dataset, total=eval_size, desc='Evaluating the eval set'):
		try:
			outputs=neural_model(category_weights, **inputs, is_training=False)
		except:
			continue
		try: ent_rels.append([infos['entity_span'], infos['relation_span'], outputs['entity']['span'], outputs['relation']['span']])
		except: ent_rels.append([infos['entity_span'], infos['relation_span'], outputs['entity']['span'], []])
		if prev_doc!=infos['document_name']:
			if doc!=0: convert(doc, full_text, '')
			full_text = ''
			doc=[infos['document_name']]
			prev_doc = infos['document_name']
			words = infos['words'].tolist()
			embeds = outputs['entity']['embedding'].tolist()
			spans = [i for i in outputs['entity']['span'] if i[-1]!=0]
			new_words, new_embeds, new_relations, pos = [words[0]], [embeds[0]], [], [(0, len(words[0]))]
			for i in range(1, len(embeds)):
				if words[i]!=words[i-1]:
					new_words.append(words[i])
					new_embeds.append(embeds[i])
					pos.append((pos[-1][-1]+1, pos[-1][-1]+1+len(words[i])))
			try:
				relations = [i for i in outputs['relation']['span'] if i[0][-1]!=0 and i[1][-1]!=0]
				for i in relations:
					if i[0] not in spans:
						spans.append(i[0])
					if i[1] not in spans:
						spans.append(i[1])
			except:
				relations=[]
			span_map = []
			for i in spans:
				span_content = ' '.join(pd.unique(words[i[0]:i[1]]).tolist())
				new_sent = ' '.join(new_words)
				span_beg = new_sent.find(span_content)
				span_end = span_beg + len(span_content)
				span_map.append((span_beg, span_end, list(entity_encode.keys())[list(entity_encode.values()).index(int(i[-1]))]))
			ents = []
			prev_embed = new_embeds[0]
			begin = 0
			if len(new_embeds)==1: continue
			for i in range(1, len(new_embeds)):
				if new_embeds[i]!=prev_embed:
					end=i-1
					if prev_embed!=0:
						st=len(' '.join(new_words[:begin]))+1-(begin==0)
						fin = len(' '.join(new_words[:end+1]))
						ent_conf=1.0
						ents.append([st, fin, ' '.join(new_words)[st:fin], list(entity_encode.keys())[list(entity_encode.values()).\
                                                                                    index(int(prev_embed))], ent_conf])
					begin=i
					prev_embed = new_embeds[i]
			if len(new_embeds)>=2:
				if new_embeds[i]!=0:
					st = len(' '.join(new_words[:begin]))+1-(begin==0)
					fin = len(' '.join(new_words))
					ent_conf=1.0
					ents.append([st, fin, ' '.join(new_words)[st:fin], list(entity_encode.keys())[list(entity_encode.values()).\
                                                                                   index(int(new_embeds[i]))], ent_conf])
			rels=[]
			for i in relations:
				left = [span_map[spans.index(i[0])][0], span_map[spans.index(i[0])][1]]
				right = [span_map[spans.index(i[1])][0], span_map[spans.index(i[1])][1]]
				rel_name = list(relation_encode.keys())[list(relation_encode.values()).index(int(i[-1]))]
				rels.append(left+right+[rel_name]+[' '.join(new_words)[left[0]:left[1]]]+[' '.join(new_words)[right[0]:right[1]]]+\
                [span_map[spans.index(i[0])][-1]]+[span_map[spans.index(i[1])][1]])
			doc.append([' '.join(new_words), pos, ents, rels])
			prev_len=len(' '.join(new_words))
			full_text+=' '.join(new_words)

		else:
			prev_doc = infos['document_name']
			words = infos['words'].tolist()
			embeds = outputs['entity']['embedding'].tolist()
			spans = [i for i in outputs['entity']['span'] if i[-1]!=0]

			for i in range(1, len(embeds)):
				if words[i]!=words[i-1]:
					new_words.append(words[i])
					new_embeds.append(embeds[i])
					pos.append((pos[-1][-1]+1, pos[-1][-1]+1+len(words[i])))
			try:
				relations = [i for i in outputs['relation']['span'] if i[0][-1]!=0 and i[1][-1]!=0]
				for i in relations:
					if i[0] not in spans:
						spans.append(i[0])
					if i[1] not in spans:
						spans.append(i[1])
			except:
				relations=[]

			span_map = []
			for i in spans:
				span_content = ' '.join(pd.unique(words[i[0]:i[1]]).tolist())
				new_sent = ' '.join(new_words)
				span_beg = new_sent.find(span_content)
				span_end = span_beg + len(span_content)
				span_map.append((span_beg, span_end, list(entity_encode.keys())[list(entity_encode.values()).index(int(i[-1]))]))

			ents=[]
			prev_embed=new_embeds[0]
			begin=0
			for i in range(1, len(new_embeds)):
				if new_embeds[i]!=prev_embed:
					end=i-1
					if prev_embed!=0:
						st=len(' '.join(new_words[:begin]))+1-(begin==0)
						fin=len(' '.join(new_words[:end+1]))
						ent_conf = 1.0
						ents.append([st+prev_len, fin+prev_len, ' '.join(new_words)[st:fin], list(entity_encode.keys())\
                   [list(entity_encode.values()).index(int(prev_embed[i]))], ent_conf])
					begin=i
					prev_embed = new_embeds[i]
			if len(new_embeds)>=2:
				if new_embeds[i]!=0:
					st = len(' '.join(new_words[:begin]))+1-(begin==0)
					fin = len(' '.join(new_words))
					ent_conf=1.0
					ents.append([st+prev_len, fin+prev_len, ' '.join(new_words)[st:fin], list(entity_encode.keys())\
                  [list(entity_encode.values()).index(int(new_embeds[i]))], ent_conf])
			rels=[]
			for i in relations:
				left=[span_map[spans.index(i[0])][0]+prev_len, span_map[spans.index(i[0])][1]+prev_len]
				right = [span_map[spans.index(i[1])][0]+prev_len, span_map[spans.index(i[1])][1]+prev_len]
				rel_name = list(relation_encode.keys())[list(relation_encode.values()).index(int(i[-1]))]
				rels.append(left+right+[rel_name] + [' '.join(new_words)[left[0]-prev_len:left[1]-prev_len]]+[' '.join(new_words)[right[0]-prev_len:right[1]-prev_len]]+[span_map[spans.index(i[0])][-1]]+[span_map[spans.index(i[1])][-1]])
			doc.append([' '.join(new_words), pos, ents, rels])
			prev_len+=len(' '.join(new_words))
			full_text+=' '.join(new_words)
	convert(doc, full_text, '')


category_weights=torch.FloatTensor([1]*len(range(entity_types)))
entity_label_map = {v:k for k, v in entity_encode.items()}
entity_classes = list(entity_label_map.keys())
entity_classes.remove(0)

relation_label_map = {v:k for k, v in relation_encode.items()}
relation_classes = list(relation_label_map.keys())
relation_classes.remove(0)
get_results((train_dataset+test_dataset), neural_model, category_weights)


def predict(neural_model, sentences):
  entity_weights = torch.tensor([1.0]*len(range(entity_types)))
  for sentence in sentences:
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
    print('Sentence: ', sentence)
    print('Entities: (', len(pred_entity_span), ')')
    for begin, end, entity_type in pred_entity_span:
      print(entity_label_map[entity_type], '|', ' '.join(tokens[begin:end]))
    print('Relations: (', len(pred_relation_span), ')')
    for e1, e2, relation_type in pred_relation_span:
      print(relation_label_map[relation_type], '|', ' '.join(tokens[e1[0]:e1[1]]), ' '.join(tokens[e2[0]:e2[1]]))

sentences="One of the major problems one is faced with when decomposing words into their constituent parts is ambiguity: the generation of multiple analyses for one input word, many of which are implausible. In order to deal with ambiguity, the MORphological PArser MORPA is provided with a probabilistic context-free grammar-LRB-PCFG-RRB-, i.e. it combines a`` conventional'' context-free morphological grammar to filter out ungrammatical segmentations with a probability-based scoring function which determines the likelihood of each successful parse. Consequently, remaining analyses can be ordered along a scale of plausibility. Test performance data will show that a PCFG yields good results in morphological parsing. MORPA is a fully implemented parser developed for use in a text-to-speech conversion system."

predict(neural_model, sent_tokenizer.tokenize(sentences))

