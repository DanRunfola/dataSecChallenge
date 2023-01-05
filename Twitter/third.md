This guide provides a basic implementation of Binary Classification Model using PyTorch. Before writing any python script we need to setup a conda Environment and instal all necessary libraries we need.

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
 
 Note that i am using BERT-12 Layer algorithm for classification but you can try with any algorithm you want.
 
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
RD=pd.read_csv("/home/rohith/pythonfiles/FullRNC.csv")
DD=pd.read_csv("/home/rohith/pythonfiles/FullDNC.csv")
MD=pd.read_csv("/home/rohith/pythonfiles/Disdata.csv")
D1=RD.iloc[:,8:9]
D1["Y"]="Original"
D1=D1.head(48000)
D2=DD.iloc[:,8:9]
D2["Y"]="Original"
D2=D2.head(48000)
D3=MD.iloc[:,8:9]
D3["Y"]="Fake"
D3=D3.head(96000)
frames=[D1, D2, D3]
Data=pd.concat(frames)
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
sentences = traindata2.clean_text.values
labels = traindata2.Y.values
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
        

 
 
 
 
 
 
 
 
 
 ```
 
