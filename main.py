from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

f = open('data.json', encoding="utf8")
dataset = json.load(f)
f.close()

data = dataset['Data']
answers = []
questions = []
for i in data:
  answers.append(i['Answer'])
  questions.append(i['Question'])
  
import re
triple_answer = []
for item in answers:
  triple_answer.append(item)

triple_question = []
for item in questions:
  cleaned_sentence = re.sub(r' của| là gì| \?| có những| nào| có bao nhiêu| nằm', '', item[0])
  triple_question.append(cleaned_sentence)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(triple_question)
question_sequences = tokenizer.texts_to_sequences(triple_question)

# Pad sequences to have the same length
max_sequence_length = max([len(seq) for seq in question_sequences])

answer_indices = {answer: i for i, answer in enumerate(answers)}
num_classes = len(answers)
answer_sequences = [answer_indices[answer] for answer in answers]

loaded_model = keras.models.load_model("my_model.h5")

app = FastAPI()

class text(BaseModel):
  Text: str


@app.post('/fetch')
async def label_location(item: text):
    question_input = []
    question_input.append(item.Text)
    cleaned_sentence = re.sub(r' của| là gì| \?| có những| nào| có bao nhiêu', '', question_input[0])
    print(cleaned_sentence)
    new_question_sequences = tokenizer.texts_to_sequences([cleaned_sentence])
    padded_new_question_sequences = pad_sequences(new_question_sequences, maxlen=max_sequence_length, padding='post')
    predict_x = loaded_model.predict(padded_new_question_sequences)
    predicted_answer_indices = np.argmax(predict_x, axis=1)

    predicted_answers = [list(answer_indices.keys())[list(answer_indices.values()).index(index)] for index in predicted_answer_indices]
    
    for question, answer in zip(question_input, predicted_answers):
        return {"result": answer}