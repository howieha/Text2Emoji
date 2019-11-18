import tweepy
import json
import pandas as pd
from tqdm import tqdm

# Get tweet text by id
def getApi():
    CONSUMER_KEY = 'dBW3AJKhfoXSlBy4einQhC1dv'
    CONSUMER_SECRET = '1FDmumY1RQUZIVSnFPLKaF4sFRIZomVPL925FMskB7mCMHA2U8'
    OAUTH_TOKEN = '1194665858565410817-k7DGha9tfOuvbakfry6TGMVjngoDab'
    OAUTH_TOKEN_SECRET = 'Jbd63HB9Qq2UDZWqAckYkVLh0dzfQeIHjtnpkcmkHkNz2'
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
    api = tweepy.API(auth)
    return api

def getText(api, tweet_id):


    tweet = api.get_status(tweet_id)
    text = tweet.text
    return text

def readData(api, fileName):
    data = pd.read_csv(fileName, sep='\t', header=[0])
    newData = pd.DataFrame(columns=['ID', 'Text', 'Emoji'])
    for i in tqdm(range(len(data.id))):
        tempRow = data.iloc[i]
        try:
            tempText = getText(api, tempRow.id)
        except:
            continue
        newData = newData.append({'ID' : tempRow.id , 'Text' : tempText, 'Emoji': tempRow.annotations.split(',')}, ignore_index=True)

    print(len(newData.ID))



if __name__ == "__main__":
    # getText(741585205367087109)
    api = getApi()
    readData(api, '../data/full_test_plaintext.txt')
    # read_json()