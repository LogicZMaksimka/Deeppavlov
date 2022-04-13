from deeppavlov import build_model
import json


model = build_model('./coqa_generative_qa_infer.json', download=False)
input_path = "/home/admin/.deeppavlov/downloads/coqa/coqa_max_tok_50.json"
output_path = "./tmp"

with open(output_path, 'w') as output_file:
    with open(input_path, 'r') as dataset_file:
        coqa = json.load(dataset_file)
        for [[question, [context]], answer] in coqa["valid"][:100]:
            res = model([question], [[context]])
            output_file.write(f"Question = {question} \n Model Answer = {res} \n Actual answer = {answer} \n Context = {context}\n\n\n")

