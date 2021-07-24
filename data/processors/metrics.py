import collections
import json
import logging
import math
import re
import string
from tqdm import tqdm
from transformers.tokenization_bert import BasicTokenizer


#   This code has been very heavily adapted/used for our CoQA model from the hugging face's Bert implmentation on SQuaD dataset. 
#   https://github.com/huggingface/transformers/blob/master/src/transformers/data/metrics/squad_metrics.py

def get_predictions(all_examples, all_features, all_results, n_best_size, max_answer_length, do_lower_case, output_prediction_file, verbose_logging, tokenizer):
    
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple("PrelimPrediction", ["feature_index", "start_index", "end_index", "score", "cls_idx",])

    all_predictions = []
    all_nbest_json = collections.OrderedDict()
    for (example_index, example) in enumerate(tqdm(all_examples, desc="Writing preditions")):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        

        score_yes, score_no, score_span, score_unk = -float('INF'), -float('INF'), -float('INF'), float('INF')
        min_unk_feature_index, max_yes_feature_index, max_no_feature_index, max_span_feature_index = -1, -1, -1, -1
        

        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            
            feature_yes_score, feature_no_score, feature_unk_score = \
                result.yes_logits[0] * 2, result.no_logits[0] * 2, result.unk_logits[0] * 2
            start_indexes, end_indexes = _get_best_indexes(result.start_logits, n_best_size), \
                                         _get_best_indexes(result.end_logits, n_best_size)

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    feature_span_score = result.start_logits[start_index] + result.end_logits[end_index]
                    prelim_predictions.append(_PrelimPrediction(feature_index=feature_index,start_index=start_index,end_index=end_index,score=feature_span_score,cls_idx=3))

            if feature_unk_score < score_unk:  
                score_unk = feature_unk_score
                min_unk_feature_index = feature_index
            if feature_yes_score > score_yes: 
                score_yes = feature_yes_score
                max_yes_feature_index = feature_index
            if feature_no_score > score_no: 
                score_no = feature_no_score
                max_no_feature_index = feature_index
                
        #including yes/no/unknown answers in preliminary predictions.
        prelim_predictions.append(_PrelimPrediction(feature_index=min_unk_feature_index,start_index=0,end_index=0,score=score_unk,cls_idx=2))
        prelim_predictions.append(_PrelimPrediction(feature_index=max_yes_feature_index,start_index=0,end_index=0,score=score_yes,cls_idx=0))
        prelim_predictions.append(_PrelimPrediction(feature_index=max_no_feature_index,start_index=0,end_index=0,score=score_no,cls_idx=1))
        prelim_predictions = sorted(prelim_predictions,key=lambda p: p.score,reverse=True)

        _NbestPrediction = collections.namedtuple("NbestPrediction", ["text", "score", "cls_idx"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            #   free-form answers (ie span answers)
            if pred.cls_idx == 3:
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = tokenizer.convert_tokens_to_string(tok_tokens)
                # removing whitespaces
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
                nbest.append(_NbestPrediction(text=final_text,score=pred.score,cls_idx=pred.cls_idx))
                
            #   'yes'/'no'/'unknown' answers
            else:
                text = ['yes', 'no', 'unknown']
                nbest.append(_NbestPrediction(text=text[pred.cls_idx], score=pred.score, cls_idx=pred.cls_idx))

        if len(nbest) < 1:
            nbest.append(_NbestPrediction(text='unknown', score=-float('inf'), cls_idx=2))

        assert len(nbest) >= 1

        probs = _compute_softmax([p.score for p in nbest])

        nbest_json = []

        for i, entry in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["score"] = entry.score
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        _id, _turn_id = example.qas_id.split()
        all_predictions.append({
            'id': _id,
            'turn_id': int(_turn_id),
            'answer': confirm_preds(nbest_json)})
        all_nbest_json[example.qas_id] = nbest_json
    #   Writing all the predictions in the predictions.json file in the BERT directory
    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        return orig_text
    
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text

def confirm_preds(nbest_json):
    #unsuccessful attempt at trying to predict for how many and True or false type of questions
    subs = [ 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine','ten', 'eleven', 'twelve', 'true', 'false']
    ori = nbest_json[0]['text']
    if len(ori) < 2:  
        for e in nbest_json[1:]:
            if _normalize_answer(e['text']) in subs:
                return e['text']
        return 'unknown'
    return ori

def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
