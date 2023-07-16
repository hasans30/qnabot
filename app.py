import os
import requests
from twilio.twiml.messaging_response import MessagingResponse

from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for )

app = Flask(__name__)

@app.route('/message', methods=['POST'])
def message():
    incoming_msg = request.values.get('Body', '').lower()
    resp = MessagingResponse()
    msg = resp.message()
    responded = False
    if 'quote' in incoming_msg:
        # return a quote
        r = requests.get('https://api.quotable.io/random')
        if r.status_code == 200:
            data = r.json()
            quote = f'{data["content"]} ({data["author"]})'
        else:
            quote = 'I could not retrieve a quote at this time, sorry.'
        msg.body(quote)
        responded = True
    if 'cat' in incoming_msg:
        # return a cat pic
        msg.media('https://cataas.com/cat')
        responded = True
    if not responded:
        msg.body('I only know about famous quotes and cats, sorry!')
    return str(resp)

  
@app.route('/health', methods=['GET'])
def health():
    return "ok"

if __name__ == '__main__':
   app.run()
