from scipy import sparse
import h5py
from keras.models import Sequential, Model, load_model
from keras.layers.core import Dense, Activation, Merge, Dropout, Reshape
from keras.layers.recurrent import LSTM
from keras.layers import merge
from keras.layers import Embedding, Input
from os.path import exists as file_exists
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

f = h5py.File('hidden_oxford_mscoco.h5','r', encoding='bytes')
data = f['hidden7_data'][:]
shape = f['hidden7_shape'][:]
indices = f['hidden7_indices'][:]
ptr = f['hidden7_indptr'][:]
mat = sparse.csr_matrix((data, indices, ptr), shape = shape)
img_array = mat.toarray()
model_save_file = 'imageqa-model.model'

vocab_dict = np.load('cocoqa/' + 'vocab-dict.npy', encoding='bytes')

quest_dict = vocab_dict[0]
quest_word_arr = vocab_dict[1]
ans_dict = vocab_dict[2]
ans_word_arr = vocab_dict[3]

img_dim = 4096
word_vec_dim = 300
maxlen = 15
answer_set_dim = len(ans_dict)
max_features = len(quest_dict)+1
batch_size = 20
sequential_mod = False

with open('cocoqa/' + 'imgid_dict.pkl', 'rb') as f:
  imgid_data = pickle.load(f)

with open('/home/vinay/IR_project/cocoqa-2015-05-17/train/' + 'questions.txt') as f:
  question_arr = f.readlines()

with open('/home/vinay/IR_project/cocoqa-2015-05-17/train/' + 'answers.txt') as f:
  answers_arr = f.readlines()

with open('/home/vinay/IR_project/cocoqa-2015-05-17/train/' + 'img_ids.txt') as f:
  img_ids_arr = f.readlines()

def get_emb(word, diction):
  if word in diction:
    return diction[word]
  else:
    return diction['UNK'.encode()]

img_arr = list(map(lambda x: img_array[imgid_data.index(x.strip())], img_ids_arr))
question_arr = list(map(lambda x: list(map(lambda y: get_emb(y.encode(), quest_dict), x.strip().split())), question_arr))

sequences_ques = pad_sequences(question_arr, maxlen=maxlen, dtype='int32',
                               padding='pre', truncating='pre', value=0.)
img_inp_seq = np.array(img_arr, dtype='float32')

#img_inp_seq = [[img_arr[i] for x in sequences_ques[i]] for i in range(len(sequences_ques))]

id_mat = np.identity(len(ans_dict), dtype='int32')
labels_arr = [id_mat[get_emb(x.strip().encode(), ans_dict)] for x in answers_arr]
labels_arr = np.array(labels_arr)

if sequential_mod:
  image_model = Sequential()
  image_model.add(Dense(word_vec_dim, 
                input_dim=img_dim,
                init='uniform',
                activation='linear'))
  image_model.add(Reshape((1,word_vec_dim), 
                  input_shape=(word_vec_dim,)))

  lang_model = Sequential()
  lang_model.add(Embedding(max_features, 
                           word_vec_dim,
                           input_length=maxlen,
                           dropout=0.2))

  vis_lstm_model = Sequential()
  vis_lstm_model.add(Merge([image_model, lang_model], 
                            mode='concat', 
                            concat_axis=1))

  vis_lstm_model.add(LSTM(512, 
                          return_sequences=False, 
                          input_shape=(maxlen+1, word_vec_dim)))
  vis_lstm_model.add(Dropout(0.5))
  vis_lstm_model.add(Dense(answer_set_dim, 
                           activation='softmax'))

  vis_lstm_model.compile(loss='categorical_crossentropy',
                         optimizer='rmsprop',
                         metrics=['accuracy'])

if file_exists(model_save_file):
  print("Model already exists. Loading...")
  vis_lstm_model = load_model(model_save_file)
else:
  image_input = Input(shape=(img_dim,))
  img_dense_out = Dense(word_vec_dim, 
                        activation='linear')(image_input)
  img_reshaped_out = Reshape((1,word_vec_dim), 
                            input_shape=(word_vec_dim,))(img_dense_out)

  quest_input = Input(shape=(maxlen,))
  embed_layer = Embedding(max_features, 
                          word_vec_dim, 
                          input_length=maxlen)(quest_input)

  img_and_ques = merge([img_reshaped_out, embed_layer],mode='concat',concat_axis=1)

  lstm_out = LSTM(512,
                return_sequences=False,
                input_shape=(maxlen+1, word_vec_dim))(img_and_ques)

  classify_out = Dense(answer_set_dim,
                     activation='softmax')(lstm_out)

  vis_lstm_model = Model(input=[image_input, quest_input], output=[classify_out])

vis_lstm_model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop',
                       metrics=['accuracy'])

# vis_lstm_model.fit([img_inp_seq, sequences_ques], labels_arr, batch_size=16, nb_epoch=1)

vis_lstm_model.fit([img_inp_seq[20000:50000], sequences_ques[20000:50000]], labels_arr[20000:50000], batch_size=16, nb_epoch=30)
vis_lstm_model.save(model_save_file)