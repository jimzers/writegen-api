# **Endpoint Documentation**

 > ## **/alive**
 > #### GET request Method to request data 
 > - check if API is alive at client side before calling any service method
 > - returns the status code
 
 > ## **/ready**
 > #### GET request Method to request data 
 > - return a dict from a view, it will be converted to a JSON response ( if a view returns a dict it will be turned into a JSON response.)

 > ## **/api/post-test**
 > #### POST request Method to request that a web server accepts the data enclosed in the body of the request message (POST -> a micro web framework for Python)
 > - Calls .json to get the JSON data and stores it in content
 > - return jsonify() call with content passed which will post JSON data to Flask 
 
> ## **/api/test-again**
> #### GET test - testing generation
> #### GET request Method to request data
> - Sets up gpt2, and starts tensorflow session for the ML platform
> - load gpt2 with "124M" (test) model & the tensorflow session
> - use .generate method on gpt2 and tensor flow session and store in output
> - res will then store output in a dict
> - return jsonify() call with res passed which will get JSON data to Flask 

> ## **/api/generate-fresh**
> #### Generating text without jumptart input
 > #### POST request Method to request that a web server accepts the data enclosed in the body of the request message (POST -> a micro web framework for Python)
> Parameter  | function
> ------------ | -------------
>  iterations |  number of samples the model generates. These samples are eventually combined in the end.
> min_sample_len | minimum length of chunks from the sample.
>  max_sample_len | maximum length of a sample. Make sure this stays below ~1000
>  past_context_len | number of tokens from the previous iteration to use as context for next model iteration
> - Sets up gpt2, and starts tensorflow session for the ML platform
> - load gpt2 with corresponding model from the model map & the tensorflow session
> - use .generate method on gpt2 and tensor flow session and store in output 
> - res will then store output in a dict
> - return jsonify() call with res passed which will get JSON data to Flask 

> ## **/api/generate**
> #### Generating text with jumptart input
 > #### POST request Method to request that a web server accepts the data enclosed in the body of the request message (POST -> a micro web framework for Python)
> Parameter  | function
> ------------ | -------------
>  input | the initial context / phrase / passage that the model will base its generation off. Have fun here!
> iterations |  number of samples the model generates. These samples are eventually combined in the end.
> min_sample_len | minimum length of chunks from the sample.
>  max_sample_len | maximum length of a sample. Make sure this stays below ~1000
>  past_context_len | number of tokens from the previous iteration to use as context for next model iteration
> - Sets up gpt2, and starts tensorflow session for the ML platform
> - load gpt2 with corresponding model from the model map & the tensorflow session
> - use .generate method on gpt2 and tensor flow session and store in output 
> - res will then store output in a dict
> - return jsonify() call with res passed which will get JSON data to Flask 


> GPT-2 generates synthetic text samples in response to the model being primed with an
> arbitrary input. The model is chameleon-like—it adapts to the style and content of the
> conditioning text. This allows the user to generate realistic and coherent continuations about a 
> topic of their choosing
