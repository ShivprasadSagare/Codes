# %%
import json
from qwikidata.linked_data_interface import get_entity_dict_from_api
from qwikidata.entity import WikidataItem
from tqdm import tqdm
import re
import time
import pickle
import sys
import os

# %%
global_qid_dict =dict()

# %%
lang = sys.argv[1]

os.chdir(f'{lang}')
print(os.getcwd())


# %%
with open(f'{lang}-stage-II-test.jsonl', 'r') as read_file:
    for id, line in enumerate(read_file):
        data = json.loads(line)
        qid = data['qid']
        global_qid_dict[qid] = 0

# %%
len(global_qid_dict)

# %%
global_qid_counter = {key: 3 for key in global_qid_dict.keys()}

# %%
while(len(global_qid_counter)):
    for qid in tqdm(list(global_qid_counter.keys())):
        try:
            item_dict = get_entity_dict_from_api(qid)
            item = WikidataItem(item_dict)
            global_qid_dict[qid] = item
            global_qid_counter.pop(qid)
        except:
            global_qid_counter[qid] -= 1
            if global_qid_counter[qid] == 0:
                global_qid_counter.pop(qid)
            time.sleep(0.05)

# %% [markdown]
# Couldn't serialize in JSON the wikidataItem class. So using pickle.

# %%
with open('global_qid_dict.pkl', 'wb') as write_file:
    pickle.dump(global_qid_dict, write_file)

# %%
with open('global_qid_dict.pkl', 'rb') as read_file:
    global_qid_dict = pickle.load(read_file)

# %%
bucket = set()
triples_ids_dict = {}

# %%
for key, item in tqdm(global_qid_dict.items()):
    if str(type(item)) != "<class 'qwikidata.entity.WikidataItem'>":
        continue
    triples_ids_dict[key] = []

    claim_groups = item.get_truthy_claim_groups()
    for pid, claim_group in claim_groups.items():
        for claim in claim_group:
            triple = {}
            triple['pid'] = pid
            bucket.add(pid)

            if claim.mainsnak.snak_datatype == 'wikibase-item':
                triple['object'] = claim.mainsnak.datavalue.value['id']
                bucket.add(claim.mainsnak.datavalue.value['id'])
                triple['object_type'] = 'wikibase-item'

            if claim.mainsnak.snak_datatype == 'quantity':
                triple['object'] = claim.mainsnak.datavalue.value['amount']
                
                if re.search('http://www.wikidata.org/entity/(Q.*)', claim.mainsnak.datavalue.value['unit']):
                    triple['unit'] = re.search('http://www.wikidata.org/entity/(Q.*)', claim.mainsnak.datavalue.value['unit']).group(1)
                    bucket.add(triple['unit'])
                else:
                    triple['unit'] = claim.mainsnak.datavalue.value['unit']
                
                triple['object_type'] = 'quantity'

            if claim.mainsnak.snak_datatype == 'time':
                triple['object'] = claim.mainsnak.datavalue.get_parsed_datetime_dict()
                triple['object_type'] = 'time'

            if claim.mainsnak.snak_datatype == 'string':
                triple['object'] = claim.mainsnak.datavalue.value
                triple['object_type'] = 'string'

            triples_ids_dict[key].append(triple)

# %%
with open('bucket.json', 'w') as write_file:
    json.dump(list(bucket), write_file) #storing list as set is not JSON serializable

# %%
with open('bucket.json', 'r') as read_file:
    bucket = json.load(read_file)

# %%
with open('triples_ids_dict.json', 'w') as write_file:
    json.dump(triples_ids_dict, write_file)

# %%
with open('triples_ids_dict.json', 'r') as read_file:
    triples_ids_dict = json.load(read_file)

# %%
len(bucket)

# %%
labels_alias_dict = {}

# %%
with open('../entities.jsonl', 'r') as read_file:
    for line in tqdm(read_file):
        data = json.loads(line)
        if data['id'] in bucket:
            labels_alias_dict[data['id']] = {'label': data['name'], 'aliases': data['aliases']}
            bucket.remove(data['id'])

# %%
len(bucket), len(labels_alias_dict)

# %%
bucket_dict = {qid: 2 for qid in bucket}

# %%
while(len(bucket_dict)):
    for qid in tqdm(list(bucket_dict.keys())):
        try:
            item_dict = get_entity_dict_from_api(qid)
            item = WikidataItem(item_dict)
            labels_alias_dict[qid] = {'label': item.get_label(), 'aliases': item.get_aliases()}
            bucket_dict.pop(qid)
        except:
            bucket_dict[qid] -= 1
            if bucket_dict[qid] == 0:
                bucket_dict.pop(qid)
        time.sleep(0.05)

# %%
with open('labels_aliases_dict.json', 'w') as write_file:
    json.dump(labels_alias_dict, write_file)

# %%
with open('labels_aliases_dict.json', 'r') as read_file:
    labels_alias_dict = json.load(read_file)

# %%
month_dict = {0: '', 1:'January', 2:'February', 3:"March", 4:'April', 5:'May', 6:'June', 7:'July', 8:'August', 9:'September', 10:'October', 11:'November', 12:'December'}

# %%
with open(f'{lang}-stage-II-test.jsonl', 'r') as read_file, open('hi-selected-triples.jsonl', 'w') as write_file:
    for line in read_file:
        data = json.loads(line)
        sentence = data['translated_sentence']

        try:
            triples = triples_ids_dict[data['qid']]
        except:
            print("qid not in triples_id_dict. skipping this instance.")
            continue

        data['candidate_triples_size'] = len(triples)

        data['matched_triples'] = []
        for triple in triples:
            if len(triple.keys()) == 1:
                continue

            # handling wikibase-item
            if triple['object_type'] == 'wikibase-item':
                try:
                    label = str(labels_alias_dict[triple['object']]['label'])
                    aliases = list(labels_alias_dict[triple['object']]['aliases'])
                except:
                    print('label, aliases not fetched for id: ', triple['object'], 'so skipping it.')
                    continue
                
                aliases.append(label)
                for alias in aliases:
                    if alias in sentence:
                        temp = []
                        try:
                            temp.append(labels_alias_dict[triple['pid']]['label'])
                        except:
                            print('predicate label not found. skipping the triple')
                            break
                        temp.append(label)
                        data['matched_triples'].append(temp)
                        break
            
            # handling quantity
            if triple['object_type'] == 'quantity':
                magnitude = triple['object'][1:]
                if triple['unit'] != '1':
                    try:
                        aliases = list(labels_alias_dict[triple['unit']])
                        aliases = [magnitude + alias for alias in aliases]
                    except:
                        print('unit id not found in labels dict. setting to empty list of aliases')
                        aliases = [magnitude]
                else:
                    aliases = [magnitude]
                
                for alias in aliases:
                    if alias in sentence:
                        temp = []
                        try:
                            temp.append(labels_alias_dict[triple['pid']]['label'])
                        except:
                            print('predicate label not found. skipping the triple')
                            break
                        temp.append(magnitude)
                        data['matched_triples'].append(temp)
                        break
            
            # handling time
            if triple['object_type'] == 'time':
                year = str(triple['object']['year'])
                month = str(month_dict[triple['object']['month']])
                day = str(triple['object']['day'])
                aliases = [day + ' ' + month + ' ' + year, month + ' ' + year, year]
                
                for alias in aliases:
                    if alias in sentence:
                        temp = []
                        try:
                            temp.append(labels_alias_dict[triple['pid']]['label'])
                        except:
                            print('predicate label not found. skipping the triple')
                            break
                        temp.append(day+' '+month+' '+year if day!=0 else month+' '+year)
                        data['matched_triples'].append(temp)
                        break
            
            # handling string
            if triple['object_type'] == 'string':
                alias = triple['object']
                if alias in sentence:
                    temp = []
                    try:
                        temp.append(labels_alias_dict[triple['pid']]['label'])
                    except:
                        print('predicate label not found. skipping the triple')
                        break
                    temp.append(alias)
                    data['matched_triples'].append(temp)
                    break

        write_file.writelines(json.dumps(data, ensure_ascii=False) + '\n')

# %%
trues, preds = [], []
with open('hi-selected-triples.jsonl', 'r') as read_file:
    for line in read_file:
        data = json.loads(line)

        lst_a = set((data['facts'][id][0], data['facts'][id][1]) for id in data['fact_index'])
        lst_b = set(list(tuple(item) for item in data['matched_triples']))
        union = lst_a.union(lst_b)
        
        true = [id for id, triple in enumerate(union) if triple in lst_a]
        pred = [id for id, triple in enumerate(union) if triple in lst_b]
        trues.append(true)
        preds.append(pred)

# %%
class AlignmentEvaluation:
    def __init__(self):
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0
        self.correct_count = 0
        self.precision_list = []
        self.recall_list = []
        self.total_count = 0
    
    def add(self, true_y: list, pred_y: list):
        '''
        Arguments
            true_y: list containing gold fact indexes
            pred_y: list containing predicted facts indexes
        '''
        set_a = set(true_y)
        set_b = set(pred_y)
        
        tp = len(set_a.intersection(set_b))
        fp = len(set_b.difference(set_a))
        fn = len(set_a.difference(set_b))

        # storing data for calculating the accuracy
        if fp==0 and fn==0:
            self.correct_count+=1

        # storing data for calculation of global precision & recall
        self.true_positive += tp
        self.false_positive += fp
        self.false_negative += fn
        self.total_count+=1

        # calulating the local precision recall
        precision = float(tp)/(len(set_b)+1e-9)
        recall = float(tp)/(len(set_a)+1e-9)            
        self.precision_list.append(precision)
        self.recall_list.append(recall)
    
    def addlist(self, true_y: list, pred_y: list):
        '''
        Arguments
            true_y: list of lists, each list contains the gold facts
            pred_y: list of lists, eacg list contains the predicted facts indexes
        '''
        assert len(true_y)==len(pred_y), "length mismatch betweent the prediction list and gold label list"

        for x, y in zip(true_y, pred_y):
            self.add(x, y)
    
    def get_scores(self):
        global_precision = float(self.true_positive) / (self.true_positive + self.false_positive + 1e-9)
        global_recall = float(self.true_positive) / (self.true_positive + self.false_negative + 1e-9)
        global_f1 = (2*global_recall*global_precision) / (global_precision+global_recall+1e-9)

        results = {
            'precision': global_precision,
            'recall': global_recall,
            'f1': global_f1,
            'avg_precision': sum(self.precision_list)/float(self.total_count+1e-9),
            'avg_recall': sum(self.recall_list)/float(self.total_count+1e-9),
            'accuracy': self.correct_count / float(self.total_count+1e-9),
            'total_count': self.total_count,
        }

        return results


# %%
eval = AlignmentEvaluation()
eval.addlist(trues, preds)
scores = eval.get_scores()
with open('hi-scores.json', 'w') as write_file:
    json.dump(scores, write_file)

# %%



