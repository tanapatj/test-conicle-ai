from oauth2client.service_account import ServiceAccountCredentials
from gcloud import storage

SERVICE_ACCOUNT_FILE = 'configs/conicle-ai.json'

credentials = ServiceAccountCredentials.from_json_keyfile_name(
    SERVICE_ACCOUNT_FILE)
storage_client = storage.Client(credentials=credentials, project='conicle-ai')
bucket = storage_client.get_bucket('conicle-ai-conicle-x-audio')

######### TRANSCRIPT BUCKET #########
TRANSCRIPT_BUCKET_NAME = 'conicle-ai-conicle-x-transcripts'
prefix = 'data/transcripts/'
dl_dir = 'transcripts/'

credentials = ServiceAccountCredentials.from_json_keyfile_name(
    'configs/conicle-ai.json')
storage_client = storage.Client(credentials=credentials, project='conicle-ai')
bucket = storage_client.get_bucket(TRANSCRIPT_BUCKET_NAME)
blobs = bucket.list_blobs(prefix=prefix)  # Get list of files
print(blobs)
for blob in blobs:
    print(blob)
    filename = blob.name.replace('/', '_')
    blob.download_to_filename(dl_dir + filename)  # Download