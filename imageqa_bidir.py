## IMAGE QUESTION ANSWERING
## Using neural networks and visual semantic
## embeddings to answer questions on images
## without intermediate stages such as
## object detection and image segmentation.

## Authors: Vinay M, Vinay B, Chetan R


##########################################################################################
## IMPORTS
##########################################################################################

from scipy import sparse
import h5py
from keras.models import Sequential, Model, load_model
from keras.layers.core import Dense, Activation, Merge, Dropout, Reshape, Flatten
from keras.layers.recurrent import LSTM
from keras.layers import merge
from keras.layers import Embedding, Input
from os.path import exists as file_exists
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2
import pickle
from keras import backend

##########################################################################################
## HYPERPARAMETERS
##########################################################################################

## Dimensions of the feature vector 
## generated from image
img_dim = 4096

## Dimensions of word vectors
## generated from questions,answers
## 300 dimentional vector
word_vec_dim = 300

## Length of the maximum sequence 
## to pass to the Recurrent
## Neural Network
## Effectively length of maximum
## possible num. of words in
## sentence/question
maxlen = 15

## Inputs are batched together
## during training, this
## parameter specifies the
## batch size
batch_size = 20

## Save file to save the model
model_save_file = 'imageqa-bidir-model.model'

## File to store data 
## after preprocessing
data_preprocess_file = 'imageqa-data-preprocess.h5'

## File to store final created
## dataset
dataset_prep_file = 'imageqa-dataset-prep.h5'

##########################################################################################
## Data Preprocessing
##########################################################################################

## Function to map words to their
## IDs from a dictionary
## Maps to Unknown token when
## faced with a word not in 
## dictionary
###------- Inputs
## word: <class 'str'>
##      the word we need the
##      id of
## diction: <class 'dict'>
###------- Outputs
## Function outputs the id
## of type <class 'int'>
def get_emb(word, diction):
  if word in diction:
    return diction[word]
  else:
    return diction['UNK'.encode()]

## Function for data preprocessing
## Also caches the processed data
## into file if file alreadt does
## not exist.
## Returns a tuple containing
## img_array -- array of the input images
## quest_dict -- dictionary of question
##               words to their IDs
## quest_word_arr -- array of words
##                   in the questions
##                   words can be
##                   indexed by ID
## ans_dict -- dictionary of answers
##             words
## ans_word_arr -- array of answer
##                 words, can be
##                 indexed by ID
def data_preprocess():
  """Data preprocess function
  for data preprocessing and caching"""
  if file_exists(data_preprocess_file):
    with open(data_preprocess_file, 'rb+') as f:
      data_tup = pickle.load(f)
  else:
    ## Extract the sparse matrix data
    f = h5py.File('hidden_oxford_mscoco.h5','r', encoding='bytes')
    data = f['hidden7_data'][:]
    shape = f['hidden7_shape'][:]
    indices = f['hidden7_indices'][:]
    ptr = f['hidden7_indptr'][:]
    
    ## Create the matrix by passing the data
    mat = sparse.csr_matrix((data, indices, ptr), shape = shape)
    img_array = mat.toarray()
    
    ## Load the dictionary from the vocabulary
    vocab_dict = np.load('cocoqa/' + 'vocab-dict.npy', encoding='bytes')
    quest_dict = vocab_dict[0]
    quest_word_arr = vocab_dict[1]
    ans_dict = vocab_dict[2]
    ans_word_arr = vocab_dict[3]

    ## Create the tuple
    data_tup = (img_array, quest_dict, quest_word_arr, ans_dict, ans_word_arr)
    with open(data_preprocess_file, 'wb+') as f:
      pickle.dump(data_tup, f)

  return data_tup


## Extract the data from the tuple
img_array, quest_dict, quest_word_arr, ans_dict, ans_word_arr = data_preprocess()

## Dimension of the softmax classifier,
## it is effectively the number of words
## in the answer dictionary
answer_set_dim = len(ans_dict)

## Maximum number of features, or
## number of words in the question
## dictionary
max_features = len(quest_dict)+1

## Encode the question into embeddings
## and to the format that the 
## model expects
### Inputs
## question : <class 'str'>
##           the question
##           to be encoded
### Outputs
## input_ques_seq : padded seq
def encode_question(question):
  question = question.lower()
  emb_q = list(map(lambda y: get_emb(y.encode(), quest_dict), question.strip().split()))
  input_ques_seq = pad_sequences([emb_q], maxlen=maxlen, dtype='int32',
                                 padding='pre', truncating='pre', value=0.)
  return input_ques_seq

##########################################################################################
## PREPARING DATASET
##########################################################################################

## Function to prepare the dataset
## to feed to the model
## Function encodes the data 
## in the format the model expects
## Returns a tuple containing:
## -- img_inp_seq : Sequences of images
##                  which forms the
##                  training set
## -- sequences_ques : Sequences of questions
##                     passed to LSTM
## -- labels_arr : Array of labels which
##                 is used by the loss function
def prepare_dataset():
  ## Check if dataset is already prepared
  ## Dataset is effectively cached
  ## as if file exists, dataset
  ## is directly loaded.
  ## This saves considerable amount of time
  if file_exists(dataset_prep_file):
    with open(dataset_prep_file, 'rb+') as f:
      data_tup = pickle.load(f)
  else:
    with open('cocoqa/' + 'imgid_dict.pkl', 'rb') as f:
      imgid_data = pickle.load(f)

    with open('/home/vinay/IR_project/cocoqa-2015-05-17/train/' + 'questions.txt') as f:
      question_arr = f.readlines()

    with open('/home/vinay/IR_project/cocoqa-2015-05-17/train/' + 'answers.txt') as f:
      answers_arr = f.readlines()

    with open('/home/vinay/IR_project/cocoqa-2015-05-17/train/' + 'img_ids.txt') as f:
      img_ids_arr = f.readlines()
    
    ## We only have the IDs of the image, hence
    ## they need to be deferenced to get the
    ## actual images
    img_arr = list(map(lambda x: img_array[imgid_data.index(x.strip())], 
                                                          img_ids_arr))
    
    ## The raw questions are cleaned,
    ## removed of special symbols,
    ## tokenized and they are
    ## encoded to their ID
    question_arr = list(map(lambda x: 
                       list(map(lambda y: 
                            get_emb(y.encode(), quest_dict), 
                            x.strip().split())), question_arr))
    
    ## The question array is padded 
    ## with the Pad token
    ## Sequences longer than
    ## maxlen are clipped
    sequences_ques = pad_sequences(question_arr, maxlen=maxlen,
                                   dtype='int32', padding='pre', 
                                   truncating='pre', value=0.)
    
    ## Converted to numpy array 
    ## to feed to Keras
    img_inp_seq = np.array(img_arr, dtype='float32')
    
    ## Creating word labels using rows
    ## an identity matrix
    id_mat = np.identity(len(ans_dict), dtype='int32')
    
    ## Answers are converted to labels
    labels_arr = [id_mat[get_emb(x.strip().encode(), ans_dict)] for x in answers_arr]
    labels_arr = np.array(labels_arr)
    
    ## Final tuple is created
    data_tup = (img_inp_seq, sequences_ques, labels_arr)
    
    ## New dataset is saved to disk
    with open(dataset_prep_file, 'wb+') as f:
      pickle.dump(data_tup, f)
  return data_tup

## Unpacking the tuple to feed
## to VIS_LSTM model
img_inp_seq, sequences_ques, labels_arr = prepare_dataset()

##########################################################################################
## VGG-19 Model
##########################################################################################

## Function to create the Oxford VGG 19
## model in Keras with TensorFlow 
## backend
## Conv_4096 layer's output is then
## taken as feature vector to be 
## fed to the LSTM
def VGG_19(weights_path=None):
  backend.set_image_dim_ordering('th')
  model = Sequential()
  model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
  model.add(Convolution2D(64, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(64, 3, 3, activation='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(128, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(128, 3, 3, activation='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(256, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(256, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(256, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(256, 3, 3, activation='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512, 3, 3, activation='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512, 3, 3, activation='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(Flatten())
  model.add(Dense(4096, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(4096, activation='relu', name="conv_4096"))
  model.add(Dropout(0.5))
  model.add(Dense(1000, activation='softmax'))

  if weights_path:
      model.load_weights(weights_path)

  return model

##########################################################################################
## Visual LSTM Model
##########################################################################################

## Visual LSTM Model
## Inputs: Question sequence
##         Image feature vector
## Output
def VIS_LSTM():
  ## FIRST Branch
  ## Layer to get Image input
  ## Shape of img_dim
  image_input = Input(shape=(img_dim,))
  
  ## Dense activation layer
  ## to transform 4096-image
  ## vector to word_vec_dim 
  ## Linear activation as suggested
  ## in paper
  img_dense_out = Dense(word_vec_dim, 
                        activation='linear')(image_input)
  
  ## Transformed vector is then
  ## to be used as first word
  ## so is reshaped to join with
  ## the word vectors
  img_reshaped_out = Reshape((1,word_vec_dim), 
                            input_shape=(word_vec_dim,))(img_dense_out)
  
  ## SECOND Branch
  ## Questions will be fed here
  ## Input shape of maxlen
  ## maximum sequence length
  quest_input = Input(shape=(maxlen,))

  ## Word encodings are then
  ## embedded onto a vector space
  ## Converted to high dimensional
  ## (300) vectors
  embed_layer = Embedding(max_features, 
                          word_vec_dim, 
                          input_length=maxlen)(quest_input)
  
  ## MERGE Branch
  ## Both the branches are merged
  ## and concatenated into one vector
  # Note: do not mistake with capital M Merge
  img_and_ques = merge([img_reshaped_out, embed_layer],
                        mode='concat',
                        concat_axis=1)

  ## One of the two bidirectional
  ## LSTM units
  ## Input shape of maxlen+1 to
  ## accomodate image and question
  ## sequences
  lstm_out = LSTM(512,
                return_sequences=False,
                input_shape=(maxlen+1, word_vec_dim))(img_and_ques)
  
  ## Reverse directional LSTM
  ## which forms the second unit
  ## of the bidirectional LSTM
  bidir_lstm_out = LSTM(512,
                    return_sequences=False,
                    input_shape=(maxlen+1, word_vec_dim),
                    go_backwards=True)(img_and_ques)
  
  ## Merging both LSTM units
  bidir_merge = merge([lstm_out,bidir_lstm_out],
                      mode='concat',
                      concat_axis=-1)
  
  ## Dropout for regularization
  ## to prevent overfitting
  dropout_bidir = Dropout(0.5)(bidir_merge)

  ## Softmax classification to
  ## generate answer
  classify_out = Dense(answer_set_dim,
                     activation='softmax')(dropout_bidir)
  
  ## Attaching the enseble together
  vis_lstm_model = Model(input=[image_input, quest_input], 
                         output=[classify_out])

  return vis_lstm_model

## Loading weights from saved
## file for inference
def load_VIS_LSTM(model_save_file=model_save_file):
  if file_exists(model_save_file):
    print("Model already exists. Loading...")
    vis_lstm_model = load_model(model_save_file)
  else:
    vis_lstm_model = VIS_LSTM()
  return vis_lstm_model

## Extracting image feature
## from image
def VGG_19_feature(file_path='vgg19_weights.h5'):
  model = VGG_19(file_path)
  sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(optimizer=sgd, loss='categorical_crossentropy')
  int_m = Model(input=model.input, output=model.get_layer("conv_4096").output)
  return int_m

## Loading both models
vis_lstm_model = load_VIS_LSTM()
vgg_model = VGG_19_feature()

## Preprocess image to convert
## to format expected by VGG19
def preprocess_image(file_path):
  im = cv2.resize(cv2.imread(file_path), (224, 224)).astype(np.float32)
  im[:,:,0] -= 103.939
  im[:,:,1] -= 116.779
  im[:,:,2] -= 123.68
  im = im.transpose((2,0,1))
  im = np.expand_dims(im, axis=0)
  return im

##########################################################################################
## TRAINING FUNCTION
##########################################################################################

## Train function to train the
## VISUAL LSTM model and
## save the weights to disk
def train(model=vis_lstm_model):
  model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop',
                       metrics=['accuracy'])
  model.fit([img_inp_seq[:1000], sequences_ques[:1000]], 
            labels_arr[:1000], batch_size=16, nb_epoch=30)
  model.save(model_save_file)

## Function to predict the
## Answer given the model,
## image feature vector and
## the question
def imageqa_predict_emb(model, img_feat_vec, question):
  ques_seq = encode_question(question)
  #img_inp = np.array([img_feat_vec], dtype='float32')
  pred = model.predict([img_feat_vec, ques_seq])
  ans_id = np.argmax(pred, axis=1)[0]
  return ans_word_arr[ans_id].decode()

## Function to precit the
## answer by image id from
## the training set
def imageqa_predict(model, img_id, question):
  return imageqa_predict_emb(model, img_inp_seq[img_id], question)

## Function to predict answer
## given the VGG19 model,
## VIS-LSTM model, image
## path from disk and the
## question in form of a string
def imageqa_predict_img(vgg_model, vis_lstm_model, img_path, question):
  img_vgg = preprocess_image(img_path)
  feat_vec = vgg_model.predict(img_vgg)
  return imageqa_predict_emb(vis_lstm_model, feat_vec, question)

## Function to predict the
## answer given image path 
## and question as string
def image_question_answer(img_path, question):
  return imageqa_predict_img(vgg_model, vis_lstm_model, img_path, question)

## Function to predict the
## answer given the image
## id in the training set
## and the question
def question_answer(img_id, question):
  return imageqa_predict(vis_lstm_model, img_id, question)

if __name__ == "__main__":
  ## Training the model
  ## when file is run
  train()