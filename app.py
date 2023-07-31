from decouple import config
import openai
from twilio.twiml.messaging_response import MessagingResponse
from aiutils import query_my_question, get_qa
from azureblobutil import download_blobs

from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for )

app = Flask(__name__)
openai.api_key = config("OPENAI_API_KEY")
portNumber=config("FLASK_RUN_PORT","5000")


@app.route('/message', methods=['POST'])
def message():
    incoming_msg = request.values.get('Body', '').lower()
    whatsapp_number = request.values.get('From').split("whatsapp:")[-1]
    print(f"Chatting with this number: {whatsapp_number}")
    resp = MessagingResponse()
    msg = resp.message()
    # Call the OpenAI API to generate text with ChatGPT
    answer=query_my_question(incoming_msg)
    msg.body(answer)
    return str(resp)

@app.route('/initdata', methods=['GET'])
def initdata():
    try:
        download_blobs()
        get_qa()
        return "ok"
    except Exception as e:
        return "error initializing data"
  
@app.route('/health', methods=['GET'])
def health():
    return "ok"

if __name__ == '__main__':
    download_blobs()
    get_qa()
    app.run(port=portNumber)
    
