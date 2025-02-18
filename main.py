from flask import Flask, request, jsonify
from pythainlp.tokenize import word_tokenize
from pythainlp.tokenize import Tokenizer
from pythainlp.corpus.common import thai_words
import pythainlp

app = Flask(__name__)

@app.route('/tokenize', methods=['POST'])
def tokenize_text():
    try:
        # Get JSON data from request
        data = request.get_json()
        print(pythainlp.__version__)
        
        # Check if 'text' field exists in request
        if 'text' not in data:
            return jsonify({
                'error': 'Missing text field in request'
            }), 400
            
        # Get text from request
        text = data['text']
        
        # Tokenize text using PyThaiNLP
        icu_tokens = word_tokenize(text, engine="icu")
        newmm_tokens = word_tokenize(text, engine="newmm")
        tokens = icu_tokens + newmm_tokens
        # tokens = custom_tokenizer.word_tokenize(text)
        
        # Return tokenized result
        return jsonify({
            'original_text': text,
            'tokens': tokens
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)