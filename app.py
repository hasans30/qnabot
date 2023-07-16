import os

from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for )

app = Flask(__name__)


@app.route('/message', methods=['POST'])
def message():
    name = request.get_json()
    return name

if __name__ == '__main__':
   app.run()
