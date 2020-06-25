from flask import Flask, jsonify, request
import random
import torch
import transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id).to(device)



app = Flask(__name__)

@app.route('/')
def index():
    return "I see the test is working."

@app.route('/api/test', methods=['GET'])
def test_model():
    # encode context the generation is conditioned on
    input_str = 'Hello. This is some sample text that I found.'
    input_ids = tokenizer.encode(input_str, add_special_tokens=False, return_tensors='pt').to(device)
    model.eval()

    min_length = 300
    max_length = 1000
    num_samples = 3
    bos_token_id = random.randint(0, tokenizer.vocab_size)
    print('bos_token_id: ' + tokenizer.decode(bos_token_id))

    output = model.generate(
        input_ids=input_ids,
        bos_token_id=random.randint(0, tokenizer.vocab_size - 1),
        do_sample=True,
        top_k=50,
        top_p=0.95,
        min_length=min_length,
        max_length=max_length,
        num_return_sequences=num_samples)

    decoded_output = []
    for sample in output:
        decoded_output.append(tokenizer.decode(
            sample, skip_special_tokens=True))
    print(decoded_output[0])

@app.route('/api/test/generate_from_input', methods=['GET'])
def generate_from_input():
    # encode context the generation is conditioned on
    input_str = 'Hello. This is some sample text that I found.'
    input_ids = tokenizer.encode(input_str, add_special_tokens=False, return_tensors='pt').to(device)
    model.eval()

    min_length = 300
    max_length = 1000
    num_samples = 1
    # bos_token_id = random.randint(0, tokenizer.vocab_size)
    # print('bos_token_id: ' + tokenizer.decode(bos_token_id))

    output = model.generate(
        input_ids=input_ids,
        bos_token_id=random.randint(0, tokenizer.vocab_size - 1),
        do_sample=True,
        top_k=50,
        top_p=0.95,
        min_length=min_length,
        max_length=max_length,
        num_return_sequences=num_samples)

    decoded_output = []
    for sample in output:
        decoded_output.append(tokenizer.decode(
            sample, skip_special_tokens=True))

    res = {
        'output': decoded_output[0]
    }

    # print(decoded_output[0])
    print('test output generated')

    return jsonify(res)

@app.route('/api/test/generate_chained_input', methods=['GET'])
def generate_chained_input():
    model.eval()
    min_length = 100
    max_length = 500
    num_samples = 1
    past_context_len = 100
    iterations = 5

    generated_str = ""

    # encode context the generation is conditioned on
    input_str = 'Hello. This is some sample text that I found.'
    input_ids = tokenizer.encode(input_str, add_special_tokens=False, return_tensors='pt').to(device)

    for i in range(iterations):
        output = model.generate(
            input_ids=input_ids,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            min_length=min_length,
            max_length=max_length,
            num_return_sequences=num_samples)

        output_str = tokenizer.decode(
            output[0, :-past_context_len], skip_special_tokens=True)
        generated_str += output_str + ' '  # or '\n'

        input_ids = output[:, -past_context_len:]

    res = {
        'output': generated_str
    }

    return jsonify(res)

@app.route('/api/generate', methods=['POST'])
def gen_text():
    content = request.json

    min_length = content['min_sample_len']
    max_length = content['max_sample_len']
    # num_samples = content['num_samples_per_iter']
    num_samples = 1
    past_context_len = content['past_context_len']
    iterations = content['iterations']



    # encode the string
    input_str = content['input']

    input_ids = tokenizer.encode(input_str, add_special_tokens=False, return_tensors='pt').to(device)

    generated_str = ""

    model.eval()

    for i in range(iterations):
        output = model.generate(
            input_ids=input_ids,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            min_length=min_length,
            max_length=max_length,
            num_return_sequences=num_samples)

        output_str = tokenizer.decode(
            output[0, :-past_context_len], skip_special_tokens=True)
        generated_str += output_str + ' '  # or '\n'

        input_ids = output[:, -past_context_len:]

    res = {
        'output': generated_str
    }

    return jsonify(res)


if __name__ == '__main__':
    app.run(debug=True)
