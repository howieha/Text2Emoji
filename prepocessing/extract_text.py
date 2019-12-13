import tweepy
import json
import pandas as pd
from tqdm import tqdm
from functools import partial
from joblib import Parallel, delayed
import multiprocessing


# Get tweet text by id
def getApi():
    CONSUMER_KEY = ''
    CONSUMER_SECRET = ''
    OAUTH_TOKEN = ''
    OAUTH_TOKEN_SECRET = ''
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    return api


# def getText(api, tweet_id):
#     tweet = api.get_status(tweet_id)
#     text = tweet.text
#     print(text)
#     return text


def lookup_tweets(api, tweet_IDs):
    tweets = []
    ids = []
    tweet_count = len(tweet_IDs)
    try:
        for i in tqdm(range(int((tweet_count / 100)))): # + 1)):    # has problem if len(tweet_IDs) is a multiple of 100
            # Catch the last group if it is less than 100 tweets
            end_loc = min((i + 1) * 100, tweet_count)
            tempData = api.statuses_lookup(tweet_IDs[i * 100:end_loc])
            tweets.extend([i.text for i in tempData])
            ids.extend([i.id for i in tempData])

        data = pd.DataFrame(list(zip(ids, tweets)), columns=['ID', 'TEXT'])
        return data
    except tweepy.TweepError:
        print('Something went wrong, quitting...')


# def helperReadData(api, data, i):
#     tempRow = data.iloc[i]
#     try:
#         tempText = getText(api, tempRow.id)
#         return tempRow.id, tempText, tempRow.annotations.split(',')
#     except:
#         return None


def readData(api, fileName, outfile, nCPU=4):
    id_emoji = pd.read_csv(fileName, sep='\t', header=[0])
    id_emoji.columns = ['ID', 'EMOJI']

    id_text = lookup_tweets(api, id_emoji['ID'].tolist())

    print(len(id_emoji.ID))
    print(len(id_text.ID))

    id_text.to_csv("../data/" + outfile + ".csv", sep='\t', index=False)

    # newData = pd.DataFrame(columns=['ID', 'TEXT', 'EMOJI'])

    # tempData = Parallel(n_jobs=4)(delayed(helperReadData)(api, data, i) for i in tqdm(range(len(data.id))))

    # for i in tqdm(range(len(data.id))):
    #     tempRow = data.iloc[i]
    #     try:
    #         tempText = getText(api, tempRow.id)
    #     except:
    #         continue
    #
    # newData = newData.append({'ID': tempRow.id, 'Text': tempText, 'Emoji': tempRow.annotations.split(',')},
    #                          ignore_index=True)

    # for i in tempData:
    #     if i is not None:
    #         print(i)
    #         newData.append({'ID': i[0], 'Text': i[1], 'Emoji': i[2]},
    #                          ignore_index=True)


if __name__ == "__main__":
    api = getApi()

    # readData(api, '../data/full_test_plaintext.txt', 'full_test_id_text')
    # readData(api, '../data/full_valid_plaintext.txt', 'full_valid_id_text')
    # readData(api, '../data/full_train_plaintext.txt', 'full_train_id_text')

    # read_json()
