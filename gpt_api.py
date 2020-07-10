from flask import Flask, jsonify, request
import gpt_2_simple as gpt2
import tensorflow as tf

from flask_cors import CORS

import random

model_name = "124M"

model_map = {
    "vanilla": "124M",
    "spongebob": "spongebob"
}

app = Flask(__name__, static_folder='../writegen/build', static_url_path='/')
cors = CORS(app, resources={r"/api/*": {"origins": ["https://writegen.com", "https://www.writegen.com"]}})  # disable this if react deployed on same directory

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/alive', methods=['GET'])
def alive_check():
    return 'OK'


@app.route('/ready', methods=['GET'])
def ready_check():
    return {
        'ready': 'yeah buddy'
    }

@app.route('/api/post-test', methods=['POST'])
def post_test():
    content = request.json

    return jsonify(content)


@app.route('/api/test-again', methods=['GET'])
def get_test():
    # setup gpt2
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess, model_name=model_name)
    output = gpt2.generate(sess,
                           checkpoint_dir="models",
                           run_name=model_name,
                           length=1000,
                           prefix="<|startoftext|>",
                           temperature=0.7,
                           top_k=40,
                           top_p=0.9,
                           # nsamples=1,
                           # batch_size=1,
                           return_as_list=True
                           )

    tf.reset_default_graph()
    sess.close()

    output_str = output[0]
    res = {
        'output': output_str
    }

    return jsonify(res)


@app.route('/api/generate-fresh', methods=['POST'])
def gen_fresh():
    content = request.json

    min_length = content['min_sample_len']
    max_length = content['max_sample_len']
    num_samples = 1
    past_context_len = content['past_context_len']
    iterations = content['iterations']

    model_str = content['writer']
    model_name = model_map[model_str]

    generated_str = ""

    random_len = random.randint(min_length, max_length)
    # setup gpt2
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess, model_name=model_name)
    output = gpt2.generate(sess,
                           checkpoint_dir="models",
                           run_name=model_name,
                           length=random_len,
                           prefix="<|startoftext|>",
                           temperature=0.9,
                           top_k=50,
                           top_p=0.95,
                           # nsamples=num_samples,
                           # batch_size=num_samples,
                           return_as_list=True
                           )
    output_str = output[0]
    output_str_arr = output_str.split(' ')
    input_str = ' '.join(output_str_arr[-past_context_len:])

    if iterations != 0:
        generated_str += ' '.join(output_str_arr[:-past_context_len]) + ' '  # or '\n'
    else:
        generated_str += output_str + ' '

    for i in range(iterations - 1):
        random_len = random.randint(min_length, max_length)
        output = gpt2.generate(sess,
                               checkpoint_dir="models",
                               run_name=model_name,
                               length=random_len,
                               prefix=input_str,
                               temperature=0.9,
                               top_k=50,
                               top_p=0.95,
                               # nsamples=num_samples,
                               # batch_size=num_samples,
                               return_as_list=True
                               )
        output_str = output[0]
        output_str_arr = output_str.split(' ')
        input_str = ' '.join(output_str_arr[-past_context_len:])
        # add the entire str if not last iteration, otherwise omit the starting input
        if iterations != iterations - 2:
            generated_str += ' '.join(output_str_arr[:-past_context_len]) + ' '  # or '\n'
        else:
            generated_str += output_str + ' '

    tf.reset_default_graph()
    sess.close()

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

    model_str = content['writer']
    model_name = model_map[model_str]

    # encode the string
    starting_str = content['input']

    generated_str = ""

    random_len = random.randint(min_length, max_length)
    # setup gpt2
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess, model_name=model_name)
    output = gpt2.generate(sess,
                           checkpoint_dir="models",
                           run_name=model_name,
                           length=random_len,
                           prefix=starting_str,
                           temperature=0.9,
                           top_k=50,
                           top_p=0.95,
                           # nsamples=num_samples,
                           # batch_size=num_samples,
                           return_as_list=True
                           )
    output_str = output[0]
    output_str_arr = output_str.split(' ')
    input_str = ' '.join(output_str_arr[-past_context_len:])

    if iterations != 0:
        generated_str += ' '.join(output_str_arr[:-past_context_len]) + ' '  # or '\n'
    else:
        generated_str += output_str + ' '

    for i in range(iterations - 1):
        random_len = random.randint(min_length, max_length)
        output = gpt2.generate(sess,
                               checkpoint_dir="models",
                               run_name=model_name,
                               length=random_len,
                               prefix=input_str,
                               temperature=0.9,
                               top_k=50,
                               top_p=0.95,
                               # nsamples=num_samples,
                               # batch_size=num_samples,
                               return_as_list=True
                               )
        output_str = output[0]
        output_str_arr = output_str.split(' ')
        input_str = ' '.join(output_str_arr[-past_context_len:])
        # add the entire str if not last iteration, otherwise omit the starting input
        if iterations != iterations - 2:
            generated_str += ' '.join(output_str_arr[:-past_context_len]) + ' '  # or '\n'
        else:
            generated_str += output_str + ' '

    tf.reset_default_graph()
    sess.close()

    res = {
        'output': generated_str
    }

    return jsonify(res)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
