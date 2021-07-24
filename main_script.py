import collections
import glob
import os
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import (AdamW, AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup, BertTokenizer, BertModel, BertConfig)
from data.processors.coqa import Extract_Features, Processor, Result
from data.processors.metrics import get_predictions
from transformers import BertModel, BertPreTrainedModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

#
train_file="coqa-train-v1.0.json"
predict_file="coqa-dev-v1.0.json"
output_directory="Bert"
pretrained_model="bert-base-uncased"
epochs = 1.0
evaluation_batch_size=16
train_batch_size=2
 
#   our model is adapted from the baseline model of https://arxiv.org/pdf/1909.10772.pdf

class BertBaseUncasedModel(BertPreTrainedModel):

    #   Initialize Layers for our model
    def __init__(self,config,activation='relu'):
        super(BertBaseUncasedModel, self).__init__(config)
        self.bert = BertModel(config)
        hidden_size = config.hidden_size
        self.fc=nn.Linear(hidden_size,hidden_size)
        self.linear1 =nn.Linear(hidden_size,1)
        self.linear2= nn.Linear(hidden_size,2)
        self.activation = getattr(F, activation)
        self.init_weights()

    def forward(self,input_ids,token_type_ids=None,attention_mask=None,start_positions=None,end_positions=None,rational_mask=None,cls_idx=None,head_mask=None):

        #   Bert-base outputs
        outputs = self.bert(input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask,head_mask=head_mask)
        output_vector, bert_pooled_output = outputs

        #   rational logits (rationale probability to calculate start and end logits)
        #   fc = w2 x relu(W1 x h)
        rational_logits = self.fc(output_vector)
        rational_logits = self.activation(self.linear1(rational_logits))

        #   pr = sigmoid(fc)
        rational_logits = torch.sigmoid(rational_logits)
        #   h1 = pr x outputvector-h
        output_vector = output_vector * rational_logits
        mask = token_type_ids.type(output_vector.dtype)
        rational_logits = rational_logits.squeeze(-1) * mask

        #   calculating start and end logits using FC(h1)
        start_end_logits = self.fc(output_vector)
        start_end_logits = self.activation(self.linear2(start_end_logits))
        
        start_logits, end_logits = start_end_logits.split(1, dim=-1)
        start_logits, end_logits = start_logits.squeeze(-1), end_logits.squeeze(-1)
        start_logits= start_logits * rational_logits
        end_logits =  end_logits * rational_logits

        #   fc2 = wa2 x relu(Wa1 x h1)
        attention  = self.fc(output_vector)
        attention  = (self.activation(self.linear1(attention))).squeeze(-1)

        #   a = SoftMax(fc2)
        attention = F.softmax(attention, dim=-1)
        attention_pooled_output = (attention.unsqueeze(-1) * output_vector).sum(dim=-2)
        unk_logits = self.fc(bert_pooled_output)
        unk_logits = self.activation(self.linear1(unk_logits))

        #   calculate yes and no logits using pooled-output = FC(a)
        yes_no_logits =self.fc(attention_pooled_output)
        yes_no_logits =self.activation(self.linear2(yes_no_logits))
        yes_logits, no_logits = yes_no_logits.split(1, dim=-1)

        if start_positions != None and end_positions != None:
            start_positions, end_positions = start_positions + cls_idx, end_positions + cls_idx
            start = torch.cat((yes_logits, no_logits, unk_logits, start_logits), dim=-1)
            end = torch.cat((yes_logits, no_logits, unk_logits, end_logits), dim=-1)
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            #   calculate cross entropy loss for start and end logits
            Entropy_loss = CrossEntropyLoss()
            start_loss = Entropy_loss(start, start_positions)
            end_loss = Entropy_loss(end, end_positions)
            #   Training objective: to minimize the total loss of both start and end logits
            total_loss = (start_loss + end_loss) / 2 
            return total_loss
        return start_logits, end_logits, yes_logits, no_logits, unk_logits

def convert_to_list(tensor):
    return tensor.detach().cpu().tolist()

def train(train_dataset, model, tokenizer, device):

    train_sampler = RandomSampler(train_dataset) 
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
    t_total = len(train_dataloader) // 1 * epochs

    # Preparing optimizer and scheduler
    
    optimizer_parameters = [{"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "LayerNorm.weight"])],"weight_decay": 0.01,},{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in ["bias", "LayerNorm.weight"])], "weight_decay": 0.0}]
    optimizer = AdamW(optimizer_parameters,lr=1e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=2000, num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(pretrained_model, "optimizer.pt")) and os.path.isfile(os.path.join(pretrained_model, "scheduler.pt")):
        optimizer.load_state_dict(torch.load(
            os.path.join(pretrained_model, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(
            os.path.join(pretrained_model, "scheduler.pt")))

    counter = 1
    epochs_trained = 0
    train_loss, loss = 0.0, 0.0
    model.zero_grad()
    iterator = trange(epochs_trained, int(epochs), desc="Epoch", disable=False)
    for _ in iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for i,batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = { "input_ids": batch[0],"token_type_ids": batch[1], "attention_mask": batch[2],"start_positions": batch[3],"end_positions": batch[4],"rational_mask": batch[5],"cls_idx": batch[6]}
            loss = model(**inputs)
            loss.backward()
            train_loss += loss.item()

             #   optimizing training parameters
            if (i + 1) % 1 == 0:
                optimizer.step()
                scheduler.step()  
                model.zero_grad()
                counter += 1
                #   Saving model weights every 1000 iterations
                if counter % 1000 == 0:
                    output_dir = os.path.join(output_directory, "model_weights")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    return train_loss/counter


def Write_predictions(model, tokenizer, device):
    dataset, examples, features = load_dataset(tokenizer, evaluate=True)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    #   wrtiting predictions once training is complete
    evalutation_sampler = SequentialSampler(dataset)
    evaluation_dataloader = DataLoader(dataset, sampler=evalutation_sampler, batch_size=evaluation_batch_size)
    mod_results = []
    for batch in tqdm(evaluation_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],"token_type_ids": batch[1],"attention_mask": batch[2]}
            example_indices = batch[3]
            outputs = model(**inputs)
        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            output = [convert_to_list(output[i]) for output in outputs]
            start_logits, end_logits, yes_logits, no_logits, unk_logits = output
            result = Result(unique_id=unique_id, start_logits=start_logits, end_logits=end_logits, yes_logits=yes_logits, no_logits=no_logits, unk_logits=unk_logits)
            mod_results.append(result)

    # Get predictions for development dataset and store it in predictions.json
    output_prediction_file = os.path.join(output_directory, "predictions.json")
    get_predictions(examples, features, mod_results, 20, 30, True, output_prediction_file, False, tokenizer)


def load_dataset(tokenizer, evaluate=False):
    #   converting raw coqa dataset into features to be processed by BERT   
    input_dir = "data" if "data" else "."
    if evaluate:
        cache_file = os.path.join(input_dir,"bert-base-uncased_dev")
    else:
        cache_file = os.path.join(input_dir,"bert-base-uncased_train")

    if os.path.exists(cache_file):
        print("Loading cache",cache_file)
        features_and_dataset = torch.load(cache_file)
        features, dataset, examples = (
            features_and_dataset["features"],features_and_dataset["dataset"],features_and_dataset["examples"])
    else:
        print("Creating features from dataset file at", input_dir)

        if not "data" and ((evaluate and not predict_file) or (not evaluate and not train_file)):
            raise ValueError("predict_file or train_file not found")
        else:
            processor = Processor()
            if evaluate:
                examples = processor.get_examples("data", 2,filename=predict_file, threads=1)
            else:
                examples = processor.get_examples("data", 2,filename=train_file, threads=1)

        features, dataset = Extract_Features(examples=examples,tokenizer=tokenizer,max_seq_length=512, doc_stride=128, max_query_length=64, is_training=not evaluate, threads=1)
    #   caching it in a cache file to reduce time
        torch.save({"features": features, "dataset": dataset, "examples": examples}, cache_file)
    if evaluate:
        return dataset, examples, features
    return dataset


def main():
    #   check if gpu is available to use it or not
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #   initialize configurations and tokenizer of Bert model 
    config = BertConfig.from_pretrained(pretrained_model)
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    model = BertBaseUncasedModel.from_pretrained(pretrained_model, from_tf=bool(".ckpt" in pretrained_model), config=config,cache_dir=None,)
    print(model)
    model.to(device)

    if (os.path.exists(output_directory) and os.listdir(output_directory)):
        raise ValueError("Output directory " + output_directory + " already exists, Change output_directory name")
    
    #   Loading dataset and training
    train_dataset = load_dataset(tokenizer, evaluate=False)
    train_loss = train(train_dataset, model, tokenizer, device)
    
    #   create output directory for model parameters and to write predictions
    if not os.path.exists(output_directory) :
        os.makedirs(output_directory)
              
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_directory)
    tokenizer.save_pretrained(output_directory)

    #   Loading Bert model for writing predictions
    model = BertBaseUncasedModel.from_pretrained(output_directory)
    tokenizer = BertTokenizer.from_pretrained(output_directory, do_lower_case=True)
    model.to(device)
    model_parameter_directory= [output_directory]
    for m in model_parameter_directory:
        model = BertBaseUncasedModel.from_pretrained(m) 
        model.to(device)
        Write_predictions(model, tokenizer, device)

if __name__ == "__main__":
    main()
