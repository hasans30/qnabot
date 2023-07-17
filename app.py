import os
from decouple import config
import openai
import requests
from twilio.twiml.messaging_response import MessagingResponse

from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for )

app = Flask(__name__)
openai.api_key = config("OPENAI_API_KEY")

@app.route('/message', methods=['POST'])
def message():
    incoming_msg = request.values.get('Body', '').lower()
    whatsapp_number = request.values.get('From').split("whatsapp:")[-1]
    print(f"Chatting with this number: {whatsapp_number}")
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
        
       # Call the OpenAI API to generate text with ChatGPT
    messages = [{"role": "user", "content": incoming_msg}]
    messages.append({"role": "system", "content": "You're an English teacher who has taught 100s of students grammar, idioms, vocab, basic English information, and beyond basics."})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.5
    )
    msg.body(response.choices[0].message.content)
    responded = True
    
    if not responded:
        msg.body('I only know about famous quotes and cats, sorry!')
    return str(resp)

  
@app.route('/health', methods=['GET'])
def health():
    return "ok"

if __name__ == '__main__':
   app.run()
