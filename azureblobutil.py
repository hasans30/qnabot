from __future__ import print_function
import os
import sys
from azure.storage.blob import ContainerClient, BlobServiceClient
import constants
from decouple import config


openai_api_key = config("OPENAI_API_KEY")

data_folder = constants.data_folder
def download_blobs():
    try:
        CONNECTION_STRING = config('AZURE_STORAGE_CONNECTION_STRING')
        CONTAINER_NAME = config('AZURE_STORAGE_CONTAINER_NAME')
        
        # create data folder
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

    except KeyError:
        print("AZURE_STORAGE_CONNECTION_STRING must be set.")
        sys.exit(1)

    container = ContainerClient.from_connection_string(CONNECTION_STRING, container_name=CONTAINER_NAME)

    blob_list = container.list_blobs()
    for blob in blob_list:
        pathname=os.path.join(data_folder, blob.name)
        downloader=container.download_blob(blob.name)
        downloader.readinto(open(pathname, 'wb'))
        print( f'downloaded {blob.name} \n')

if __name__ == "__main__":
    download_blobs()