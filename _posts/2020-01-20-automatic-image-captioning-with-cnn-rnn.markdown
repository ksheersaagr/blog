---
layout: post
title: "Automatic Image Captioning with CNN & RNN"    
author: krunal kshirsagar
---

Generally, a captioning model is a combination of two separate architecture that is CNN (Convolutional Neural Networks)& RNN (Recurrent Neural Networks) and in this case LSTM (Long Short Term Memory), which is a special kind of RNN that includes a memory cell, in order to maintain the information for a longer period of time. Basically, CNN is used to generate feature vectors from the spatial data in the images and the vectors are fed through the fully connected linear layer into the RNN architecture in order to generate the sequential data or sequence of words that in the end generate description of an image by applying various image processing techniques to find the patterns in an image.

# Dataset used for Training the Model
We'll be using [**MS-COCO dataset also stands for Microsoft Common Objects in COntext**](https://cocodataset.org/#download). This is an advance dataset where each image is paired with five associated captions that describes the content of that particular image. For example, If you were asked to write a caption that describes the image below, how would you do that?

<img src="{{ site.baseurl }}/images/2020-01-20-automatic-image-captioning-with-cnn-rnn-bunch-of-goblins.png">

_First, you might look at the image and take note of different objects like different people and kites and blue sky. Then based on how these objects are placed in an image and their relationship with each other, you might think that these people are flying kites. They’re in this big grassy area, so they may also be in a park. After, collecting these visual observations you could put together a phrase that describes the image as, **“People flying kite in a park”**. We use a combination of spatial observation and sequential text descriptions to write a caption, and that’s exactly how the model that uses CNN and RNN architectures rolls._

## Visualize the Dataset

```python
import os
import sys
sys.path.append('/opt/cocoapi/PythonAPI')
from pycocotools.coco import COCO

# initialize COCO API for instance annotations
dataDir = '/opt/cocoapi'
dataType = 'val2014'
instances_annFile = os.path.join(dataDir, 'annotations/instances_{}.json'.format(dataType))
coco = COCO(instances_annFile)

# initialize COCO API for caption annotations
captions_annFile = os.path.join(dataDir, 'annotations/captions_{}.json'.format(dataType))
coco_caps = COCO(captions_annFile)

# get image ids 
ids = list(coco.anns.keys())
```


```python
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
%matplotlib inline

# pick a random image and obtain the corresponding URL
ann_id = np.random.choice(ids)
img_id = coco.anns[ann_id]['image_id']
img = coco.loadImgs(img_id)[0]
url = img['coco_url']

# print URL and visualize corresponding image
print(url)
I = io.imread(url)
plt.axis('off')
plt.imshow(I)
plt.show()

# load and display captions
annIds = coco_caps.getAnnIds(imgIds=img['id']);
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)
```

<img src="{{ site.baseurl }}/images/2020-01-20-automatic-image-captioning-with-cnn-rnn-coco-dataset.png
">


## The CNN-RNN Architecture

# Encoder-Decoder:  

<img src="{{ site.baseurl }}/images/2020-01-20-automatic-image-captioning-with-cnn-rnn-encoder-decoder.png
">

End to End, we want our captioning model to take in an image as input and output a text description of that image. The input image will be processed by a CNN and will connect the output of the CNN to the input of the RNN which will allow us to generate descriptive texts.


# ResNet Architecture:  

<img src="{{ site.baseurl }}/images/2020-01-20-automatic-image-captioning-with-cnn-rnn-resnet-architecture.png
">

So, in order to generate a description, we feed a particular image into a pre-trained CNN like ResNet architecture. At the end of this network is a softmax classifier that outputs a vector of class scores but we don’t want to classify an image, instead we want a set of features that represents the spatial content in the image. To get that kind of spatial content, **we’re going to remove the final fully connected layer that classifies the image** and look at it’s earlier layer that distills the spatial information in the image.


# Encoder-CNN:

Now, we’re using the CNN as a feature extractor that compresses the huge amount of extraction contained in the original image into a smaller representation. This **CNN is often called the encoder because it encodes the content of the image into a smaller feature vector.** Then we can process this feature vector and use it as an initial input to the following RNN.

<img src="{{ site.baseurl }}/images/2020-01-20-automatic-image-captioning-with-cnn-rnn-cnn-encoder.png
">

```python
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
```


# Decoder-RNN:

The job of the RNN is to decode the process vector and turn it into a sequence of words. Thus, this portion of the network is often called a decoder.

<img src="{{ site.baseurl }}/images/2020-01-20-automatic-image-captioning-with-cnn-rnn-rnn-decoder.png
">

```python
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(input_size = embed_size,hidden_size = hidden_size,
                            num_layers = num_layers, batch_first = True)
        
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        captions = captions[:, :-1]
        embed = self.embedding_layer(captions)
        embed = torch.cat((features.unsqueeze(1), embed), dim = 1)
        lstm_outputs, _ = self.lstm(embed)
        out = self.linear(lstm_outputs)
        
        return out
```

## Loading Annotations/Tokenizing Captions:

The RNN component of the captioning network is trained on the captions in the COCO dataset. We’re aiming to train the RNN to predict the next word of a sentence based on previous words. But, how exactly can it train on string data? Neural nets do not do well with strings. They need a well defined numerical alpha to effectively perform back-propagation and learn to produce similar output. So, we have to transform the captions associated with the image into a list of tokenized words. This tokenization turns any string into a list of words.

<img src="{{ site.baseurl }}/images/2020-01-20-automatic-image-captioning-with-cnn-rnn-token.png
">

```python
sample_caption = 'A person doing a trick on a rail while riding a skateboard.'

import nltk
import torch

sample_tokens = nltk.tokenize.word_tokenize(str(sample_caption).lower())

sample_caption = []
start_word = data_loader.dataset.vocab.start_word
sample_caption.append(data_loader.dataset.vocab(start_word))
sample_caption.extend([data_loader.dataset.vocab(token) for token in sample_tokens])
end_word = data_loader.dataset.vocab.end_word
sample_caption.append(data_loader.dataset.vocab(end_word))
sample_caption = torch.Tensor(sample_caption).long()
print(sample_caption)
```


## Working of Tokenization:  

First, we iterate through all of the training captions and create a dictionary that maps all unique words to a numerical index. So, every word we come across will have a corresponding integer value that can find in this dictionary. The words in this dictionary are referred to as our vocabulary. The vocabulary typically also includes a few special tokens.


```python
# Preview the word2idx dictionary.
dict(list(data_loader.dataset.vocab.word2idx.items())[:10])
```

<img src="{{ site.baseurl }}/images/2020-01-20-automatic-image-captioning-with-cnn-rnn-dataloader-dataset.png
">

```python
# Modify the minimum word count threshold.
vocab_threshold = 4

# Obtain the data loader.
data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=False)

# Print the total number of keys in the word2idx dictionary.
print('Total number of tokens in vocabulary:', len(data_loader.dataset.vocab))
```

<img src="{{ site.baseurl }}/images/2020-01-20-automatic-image-captioning-with-cnn-rnn-total-no-of-tokens.png
">


## Embedding Layer:

**There's one more step before these words get sent as input to an RNN and thats the embedding layer, which transforms each word in a caption into a vector of a desired consistent shape.**


# Words to Vectors:

<img src="{{ site.baseurl }}/images/2020-01-20-automatic-image-captioning-with-cnn-rnn-word-2-vec.png
">

At this point, we know that you cannot directly feed words into an LSTM and expect it to be able to train or produce the correct output. These words first must be turned into a numerical representation so that a network can use normal loss functions and optimizers to calculate how “close” a predicted word and ground truth word (from a known, training caption) are? So, we typically turn a sequence of words into a sequence of numerical values; a vector of numbers where each number maps to a specific word in our vocabulary.

```python
def sample(self, inputs, states=None, max_len=20):
    " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
    output_sentence = []
    for i in range(max_len):
        lstm_outputs, states = self.lstm(inputs, states)
        lstm_outputs = lstm_outputs.squeeze(1)
        out = self.linear(lstm_outputs)
        last_pick = out.max(1)[1]
        output_sentence.append(last_pick.item())
        inputs = self.embedding_layer(last_pick).unsqueeze(1)
    
    return output_sentence
```

## Training the RNN-Decoder model with suitable parameters:

The Decoder will be made of LSTM cells which is good for remembering the lengthy sequences of words. Each LSTM cell is expecting to see the same shape of the input vector at each time-step. The very first cell is connected to the output feature vector of the CNN encoder. The input to the RNN for all future time steps will be the individual words of the training caption. So, at the start of training, we have some input from our CNN, and LSTM cell with initial state. Now the RNN has two responsibilities:  
1. To Remember spatial information from the input feature vector.
2. To Predict the next word.

We know that the very first word it produces should always be the `<start>` token and the next word should be those in the training caption. At every time step, we look at the current caption word as input and combine it with the hidden state of the LSTM cell to produce an output. This output is then passed to the fully connected layer that produces a distribution that represents the most likely next word. We feed the next word in the caption to the network and so on until we reach the `<end>`token. The hidden state of an LSTM is a function of the input token to the LSTM and the previous state also referred to as the recurrence function. The recurrence function is defined by weights and during the training process, this model uses back-propagation to update these weights until the LSTM cells learn to produce the correct next word in the caption given the current input word. As with most models, you can also take advantage of batching the training data. The model updates its weights after each training batch with the batch size is the number of image caption pairs sent through the network during a single training step. Once the model has trained, it will have learned from many image caption pairs and should be able to generate captions for new image data.

**Note: Please do play around with hyperparameters if you don’t get the desired result. I’ve nailed the hyperparameters by setting them to particular value based on instinct in one go. Also, please make sure not to change the values of mean & standard deviation in transforms.Normalize() as those values are default and are considered after rigorous training of ResNet architecture on the ImageNet Dataset.**

```python
import torch
import torch.nn as nn
from torchvision import transforms
import sys
sys.path.append('/opt/cocoapi/PythonAPI')
from pycocotools.coco import COCO
from data_loader import get_loader
from model import EncoderCNN, DecoderRNN
import math


## TODO #1: Select appropriate values for the Python variables below.
batch_size = 64          # batch size
vocab_threshold = 5        # minimum word count threshold
vocab_from_file = True    # if True, load existing vocab file
embed_size = 300           # dimensionality of image and word embeddings
hidden_size = 512          # number of features in hidden state of the RNN decoder
num_epochs = 3             # number of training epochs
save_every = 1             # determines frequency of saving model weights
print_every = 100          # determines window for printing average loss
log_file = 'training_log.txt'       # name of file with saved training loss and perplexity

# (Optional) TODO #2: Amend the image transform below.
transform_train = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

# Build data loader.
data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=vocab_from_file)

# The size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)

# Initialize the encoder and decoder. 
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

# Move models to GPU if CUDA is available. 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
decoder.to(device)

# Define the loss function. 
criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

# TODO #3: Specify the learnable parameters of the model.
params = list(decoder.parameters()) + list(encoder.embed.parameters())

# TODO #4: Define the optimizer.
optimizer = torch.optim.Adam(params = params, lr = 0.001)

# Set the total number of training steps per epoch.
total_step = math.ceil(len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size)
```


**Lowest Loss: 1.74 after more than 6 hours of training:**

<img src="{{ site.baseurl }}/images/2020-01-20-automatic-image-captioning-with-cnn-rnn-lowest-loss-6-hours-of-training.png
">

**_Do test the model on Test/Validation Data to check for overfitting as the above result is of the training set._**


## Generate Predictions:

A function **`(get_prediction)`** used to loop over images in the test dataset and print your model's predicted caption.

```python
def get_prediction():
    orig_image, image = next(iter(data_loader))
    plt.imshow(np.squeeze(orig_image))
    plt.title('Sample Image')
    plt.show()
    image = image.to(device)
    features = encoder(image).unsqueeze(1)
    output = decoder.sample(features)    
    sentence = clean_sentence(output)
    print(sentence)
```

# Output:

Call the **`(get_prediction)`** function every time you want the result.

### When the model performed better:

<img src="{{ site.baseurl }}/images/2020-01-20-automatic-image-captioning-with-cnn-rnn-result-when-model-perform-better.png
">

<img src="{{ site.baseurl }}/images/2020-01-20-automatic-image-captioning-with-cnn-rnn-result-when-model-perform-better-1.png
">

### When the model didn't perform well:

<img src="{{ site.baseurl }}/images/2020-01-20-automatic-image-captioning-with-cnn-rnn-result-when-model-not-perform-that-well.png
">

<img src="{{ site.baseurl }}/images/2020-01-20-automatic-image-captioning-with-cnn-rnn-result-when-model-not-perform-that-well-1.png
">

**Clearly, as you can see the model struggles if the image is cluttered with more objects. Hence, the model finds it difficult to generate a long sequence of words that relate to each other using the spatial data in the image.**

Make sure to check out my project on **[Github](https://github.com/Noob-can-Compile/Automatic-Image-Captioning/).**

## References:

1. [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf)

2. [Chris Olah's blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

3. [Exploring LSTMs](http://blog.echen.me/2017/05/30/exploring-lstms/)

4. [Karpathy's Blog on RNN](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

5. [RNN Slides of CS231n Lecture 10 of 2019- Fei-Fei Li](http://cs231n.stanford.edu/slides/2019/cs231n_2019_lecture10.pdf)

6. [Detection and Segmentation slides of CS231n Lecture 11 of 2017- Fei-Fei Li](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture11.pdf)

7. [RNN Cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks)

8. [Convolutional Neural Networks for Visual Recognition Spring 2017 Stanford Youtube](https://www.youtube.com/watch?v=6niqTuYFZLQ&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=10)
