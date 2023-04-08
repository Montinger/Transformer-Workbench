
######################################################
### IMPORTS                               ###
######################################################
# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.

import re
import flask
import torch
import waitress
import tokenizers

import common
from params import max_nr_tokens




######################################################
### Load Tokenizer and Model                       ###
######################################################

# Load Tokenizer
tokenizer = tokenizers.Tokenizer.from_file('results/tokenizer.json')

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
model = torch.load('results/model-own-30.pth', map_location=torch.device(device))


# Flask constructor takes the name of current module (__name__) as argument.
app = flask.Flask(__name__)



common_header = """
<!DOCTYPE html>
<html>

<!-- Bootstrap 5: Latest compiled and minified CSS -->
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">

<!-- Bootstrap 5: Latest compiled JavaScript -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>

<head>
    <title>Flask Translation App</title>

    <style>
    .bg-lightgrey {background-color: rgb(221, 221, 221);}
    .bg-lightcyan   {background-color: #aacaee;}
    .bg-lightblue   {background-color: #b7c2fa;}
    .bg-lightpurple {background-color: #c1a4f4;}
    </style>
</head>

<body>
"""
# old blue value: ; old purple: 9C6DEE


common_end = """
</body>
</html>
"""

text_input_form = """
<div class="row d-flex justify-content-center">
  <div class="col-sm-5">
    <h1>English to German Translation App</h1>
  </div>
</div>

<div class="row d-flex justify-content-center">
  <div class="col-sm-5 bg-lightgrey">
    <form action="/data" method = "POST">
      <div class="mb-3 mt-3">
        <label for="comment">English text to translate:</label>
        <textarea class="form-control" rows="5" id="comment" name="text"></textarea>
      </div>
      <button type="submit" class="btn btn-primary m-2">Translate</button>
      <button type="reset" class="btn btn-secondary m-2">Clear Input</button>
    </form>
  </div>
</div>
"""


results_display_template = """
<div class="row d-flex justify-content-center">
  <div class="col-sm-6">
    <div class="text-center">
      <h1>English to German Translation Results</h1>
      <a href="/" class="btn btn-primary active m-2">Back to Input Form</a>
    </div>
  </div>
</div>

<div class="row d-flex justify-content-center">
  <div class="col-sm-4 bg-lightpurple p-2">
    <h2>Input Text</h2>
    {{ data_io_dict['input'] }}
  </div>
</div>

<div class="row d-flex justify-content-center">
  <div class="col-sm-4 bg-lightblue p-2">
    <h3>Output Greedy Translation</h3>
    {{ data_io_dict['translated_text_greedy'] }}
  </div>
  <div class="col-sm-4 bg-lightcyan p-2">
    <h3>Output Beam Search Translation</h3>
    {{ data_io_dict['translated_text_beam'] }}
  </div>
</div>
"""

data_io_dict = {}


@app.route('/')
def translation_form():
    return common_header + text_input_form + common_end



@app.route('/data/', methods = ['POST', 'GET'])
def data():
    if flask.request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if flask.request.method == 'POST':
        form_data = flask.request.form
        data_io_dict['input'] = form_data['text']

        input_text = common.cleanse_text(data_io_dict['input'], remove_unicode=False)
        
        # The model is confused by quotations marks ending after a sentence. We just revert it to here
        input_text = input_text.replace('."', '".')

        input_sentences = common.tokenize_sentences(input_text)
        print(input_sentences)
        translated_sentences_greedy = [common.translate_text_greedy(s, model=model, tokenizer=tokenizer, device=device) for s in input_sentences]
        translated_sentences_beam = [common.translate_text_beam_search(s, model=model, tokenizer=tokenizer, device=device) for s in input_sentences]
        data_io_dict['translated_text_greedy'] = ' '.join(translated_sentences_greedy)
        data_io_dict['translated_text_beam'] = ' '.join(translated_sentences_beam)

    return flask.redirect('/results') #str(data_io_dict) #render_template('data.html',form_data = form_data)

@app.route('/results')
def display_results():
    return flask.render_template_string((common_header + results_display_template + common_end), data_io_dict=data_io_dict)


# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    ## app.run(debug=True)
    waitress.serve(app, host='0.0.0.0', port=5000)
