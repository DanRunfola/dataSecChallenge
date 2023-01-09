This guide provides a basic implementation of Binary Text Classification Model using PyTorch. Before writing any python script we need to setup a conda Environment and instal all necessary libraries we need.

# Setting Up Enviroment

It is highly recommended that you use conda environments to avoid conflicts with other packages. Assuming you have anaconda in your local computer follow the below steps to set up conda environment:

Use the following line to create your environment and type y when prompted Proceed ([y]/n)?

```
conda create -n [ENVNAME]
```
(note: in the place of [ENVNAME] you guys can give whatever the name you want eg: DLEnv)

Use the following line to activate your new environment:
```sh
conda activate [ENVNAME]
```
Use the following line to deactivate your environment:
```sh
conda deactivate
```

# Installing Libraries

We have to activate conda environment before installing any libraries. There are bunch of libraries we need to install like pandas,numpy,sklearn,nltk, pytorch and transformers.

Once you've activated your environment, you can install any packages using standard commands:
```
#For pandas,numpy and sklearn:
conda install pandas
conda install numpy
conda install scikit-learn

#For nltk:
conda install -c anaconda nltk

#For tqdm:
conda install -c conda-forge tqdm

#For pytorch:
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia(for both cpu and gpu)

#For transformers:
conda install -c huggingface transformers

```

# The Python Script
 
 Note that i am using BERT-12 Layer algorithm for classification but you can try with any algorithm you want. We have two scripts: a train script and a test script. In the train script, the model is trained using the traindataset, and in the test script, the trained model is loaded and a test dataset is provided for label prediction.
 
 # Training
 
 ```python
 
#Importing all necessary libraries
import pandas as pd
import numpy as np
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import re
from nltk.tokenize import TweetTokenizer
nltk.download
nltk.download('wordnet')
nltk.download('stopwords')

import torch
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os
import sys
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

#Loading the data
Data=pd.read_csv("Give path to the csv train dataset file you downloaded")
Data=Data.iloc[:,1:](Keeping only text and label columns)
Data

#Checking and removing null values
for col in Data.columns:
    print(col, Data[col].isnull().sum())
Data= Data.dropna()

#Now we perform some data preprocessing tasks.
#Converting text to lowercase
Data['clean_text'] = Data.full_text.str.lower()

#Removing urls
Data.clean_text = Data.clean_text.apply(lambda x: re.sub(r'https?:\/\/\S+', '', x))
Data.clean_text.apply(lambda x: re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', x)) 

#Removing non-letter characters
Data.clean_text = Data.clean_text.apply(lambda x: re.sub(r'&[a-z]+;', '', x))
Data.clean_text = Data.clean_text.apply(lambda x: re.sub(r"[^a-z\s\(\-:\)\\\/\];=\'\\#\\]", '', x))

#Removing hashtag and quote symbols
Data.clean_text = Data.clean_text.apply(lambda x: re.sub(r'#', '', x))
Data.clean_text = Data.clean_text.apply(lambda x: re.sub(r"'", '', x))

#Tokenization
tknzr = TweetTokenizer()
Data['clean_text'] = Data['clean_text'].apply(tknzr.tokenize)

#Removing Punctuations
PUNCUATION_LIST = list(string.punctuation)
def remove_punctuation(word_list):
    """Remove punctuation tokens from a list of tokens"""
    return [w for w in word_list if w not in PUNCUATION_LIST]
Data['clean_text'] = Data['clean_text'].apply(remove_punctuation)

#Removing Stopwords
stop_words = set(stopwords.words('english'))
Data['clean_text'] = Data['clean_text'].apply(lambda x: [item for item in x if item not in stop_words])

#Performing Lemmatization
lemmatizer = nltk.stem.WordNetLemmatizer()
w_tokenizer = TweetTokenizer()
def lemmatize_text(text):
 return [(lemmatizer.lemmatize(w)) for w in text]
Data['clean_text'] = Data['clean_text'].apply(lemmatize_text)

#Putting tokens back into string
Data['clean_text']=[' '.join(map(str,l)) for l in Data['clean_text']]
 
#Labeling original as '0' and Fake as '1'
Data.Y[Data.Y == 'Original'] = 0
Data.Y[Data.Y == 'Fake'] = 1
TrainData=Data.iloc[:,1:]

# Get the lists of sentences and their labels.
sentences = Traindata.clean_text.values
labels = Traindata.Y.values
labels = np.asarray(labels).astype(np.float32)

#Importing the Bert Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
 
# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []

# For every sentence...
for sent in sentences:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 64,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)


# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids, attention_masks, labels)

# Create a 90-10 train-validation split.

# Calculate the number of samples to include in each set.
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

# Divide the dataset by randomly selecting samples.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
 
# The DataLoader needs to know our batch size for training, so we specify it 
# here. For fine-tuning BERT on a specific task, the authors recommend a batch 
# size of 16 or 32.
batch_size = 32

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )
        
#Network
class BertClassifier(nn.Module):

    def __init__(self, dropout=0.1):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768,1)
        self.relu = nn.Sigmoid()

    def forward(self, b_input_ids, b_input_mask):

        _, pooled_output = self.bert(input_ids= b_input_ids, attention_mask=b_input_mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer
        
#training method

def train(model, train_data, val_data, learning_rate, epochs, train_size, val_size):

    trainlen=train_size
    vallen=val_size

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    for_print = 0

###############################################
## Only if you are using gpu
    if use_cuda:

            model = model.cuda()
            criterion = criterion.cuda()
###############################################            

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0
            model.train()

            for batch in tqdm(train_data):


                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                # this is resetting the model parameters for each batch of training
                # we want to continuously update model parameters
                # model.zero_grad()
                
                optimizer.zero_grad()
                
                output = model(b_input_ids, b_input_mask)
                train_pred_probs = torch.flatten(output)
 
                batch_loss = criterion(train_pred_probs.float(), b_labels)

                total_loss_train += batch_loss.item()

                acc = (train_pred_probs.round() == b_labels).sum().item()
                total_acc_train += acc
                 
                batch_loss.backward() 
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

            model.eval()

            with torch.no_grad():
                for batch in val_data:
                
                    b_input_ids = batch[0].to(device)
                    b_input_mask = batch[1].to(device)
                    b_labels = batch[2].to(device)


                    output = model(b_input_ids,b_input_mask)
                    val_pred_probs = torch.flatten(output)

                    batch_loss = criterion(val_pred_probs.float(), b_labels)
                    total_loss_val += batch_loss.item()
                    acc = (val_pred_probs.round() == b_labels).sum().item()
                    total_acc_val += acc
            
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / trainlen : .3f} \
                | Train Accuracy: {total_acc_train / trainlen : .3f} \
                | {total_acc_train} and {trainlen}\
                | Val Loss: {total_loss_val / vallen: .3f} \
                | Val Accuracy: {total_acc_val / vallen: .3f}\
                | {total_acc_val} and {vallen}'
                )
                  
EPOCHS = 4
model = BertClassifier()
LR = 2e-5
              
train(model,train_dataloader, validation_dataloader, LR, EPOCHS, train_size, val_size)

torch.save(model.state_dict(), 'Path to the folder where you want to save the weights')
 
 ```
 
 # Testing
 
We should carry out all data preprocessing tasks the same way you did for training data. And in test dataset there will be no labels only tweets.
 
 
 ```python
 
#Importing all necessary libraries
import pandas as pd
import numpy as np
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import re
from nltk.tokenize import TweetTokenizer
nltk.download
nltk.download('wordnet')
nltk.download('stopwords')

import torch
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os
import sys
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

#Loading the data
Data=pd.read_csv("Give path to the csv test dataset file you downloaded")
Data=Data.iloc[:,1:](Keeping only tweet)
Data

#Checking and removing null values
for col in Data.columns:
    print(col, Data[col].isnull().sum())
Data= Data.dropna()

#Now we perform some data preprocessing tasks.
#Converting text to lowercase
Data['clean_text'] = Data.full_text.str.lower()

#Removing urls
Data.clean_text = Data.clean_text.apply(lambda x: re.sub(r'https?:\/\/\S+', '', x))
Data.clean_text.apply(lambda x: re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', x)) 

#Removing non-letter characters
Data.clean_text = Data.clean_text.apply(lambda x: re.sub(r'&[a-z]+;', '', x))
Data.clean_text = Data.clean_text.apply(lambda x: re.sub(r"[^a-z\s\(\-:\)\\\/\];=\'\\#\\]", '', x))

#Removing hashtag and quote symbols
Data.clean_text = Data.clean_text.apply(lambda x: re.sub(r'#', '', x))
Data.clean_text = Data.clean_text.apply(lambda x: re.sub(r"'", '', x))

#Tokenization
tknzr = TweetTokenizer()
Data['clean_text'] = Data['clean_text'].apply(tknzr.tokenize)

#Removing Punctuations
PUNCUATION_LIST = list(string.punctuation)
def remove_punctuation(word_list):
    """Remove punctuation tokens from a list of tokens"""
    return [w for w in word_list if w not in PUNCUATION_LIST]
Data['clean_text'] = Data['clean_text'].apply(remove_punctuation)

#Removing Stopwords
stop_words = set(stopwords.words('english'))
Data['clean_text'] = Data['clean_text'].apply(lambda x: [item for item in x if item not in stop_words])

#Performing Lemmatization
lemmatizer = nltk.stem.WordNetLemmatizer()
w_tokenizer = TweetTokenizer()
def lemmatize_text(text):
 return [(lemmatizer.lemmatize(w)) for w in text]
Data['clean_text'] = Data['clean_text'].apply(lemmatize_text)

#Putting tokens back into string
Data['clean_text']=[' '.join(map(str,l)) for l in Data['clean_text']]
 
#Labeling original as '0' and Fake as '1'
Data.Y[Data.Y == 'Original'] = 0
Data.Y[Data.Y == 'Fake'] = 1
TrainData=Data.iloc[:,1:]

# Get the lists of sentences.
sentences = Traindata.clean_text.values

#Importing the Bert Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
 
# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []

# For every sentence...
for sent in sentences:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 64,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

# Combine the testing inputs into a TensorDataset.
prediction_data = TensorDataset(input_ids, attention_masks)
 
# The DataLoader needs to know our batch size for training, so we specify it 
# here. For fine-tuning BERT on a specific task, the authors recommend a batch 
# size of 16 or 32.
batch_size = 32

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
prediction_sampler = RandomSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

#Network
class BertClassifier(nn.Module):

    def __init__(self, dropout=0.1):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768,1)
        self.relu = nn.Sigmoid()

    def forward(self, b_input_ids, b_input_mask):

        _, pooled_output = self.bert(input_ids= b_input_ids, attention_mask=b_input_mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer
        
 model = BertClassifier()
 
#load the model
model.load_state_dict(torch.load('Path to the file where you saved the weights'))
        
#Predicting


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#############################################
Only if using Gpu:
if use_cuda:

    model = model.cuda()
#############################################

model.eval()

predictions = []

with torch.no_grad():
    for batch in prediction_dataloader:
    
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)

        output = model(b_input_ids,b_input_mask)
        test_pred_probs = torch.flatten(output)
        test_pred_probs.round()
        out_labels=test_pred_probs.detach().cpu().numpy()
        predictions.append(out_labels)
 
