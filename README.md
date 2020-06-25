# Sample transformer app

Hey! This is a sample transformer app I wrote to test deployment to platforms.

To setup the dependencies:

```
pip install -r requirements.txt
```

To run the server:

```
python app.py
```

Quick documentation on some endpoints:

`/api/test` : (GET) tests generation

`/api/generate` : (POST) tests user input to generate long sequence.

Parameters of `/api/generate`:

`input`: the initial context / phrase / passage that the model will base its generation off. Have fun here!

`iterations`: number of samples the model generates. These samples are eventually combined in the end.

`min_sample_len`: minimum length of chunks from the sample.

`max_sample_len`: maximum length of a sample. Make sure this stays below ~1000

`past_context_len`: number of tokens from the previous iteration to use as context for next model iteration