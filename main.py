import numpy as np
import pandas as pd
import os
import glob
import json
import datetime
import csv
import matplotlib.pyplot as plt
import random 
class tweet:



    ''' Constructor for tweet        '''

    def __init__(self,timestamp,is_source_tweet,is_rumor,source=0):
        self.timestamp = datetime.datetime.strptime(timestamp,'%a %b %d %H:%M:%S %z %Y')
        self.is_source_tweet=is_source_tweet
        self.is_rumor=is_rumor
        if source==0:
            self.diff_with_source = 0
        else:
            self.diff_with_source=self.minute- source.minute


    @property
    def minute(self):
        return self.timestamp.second / 60 + self.timestamp.minute + self.timestamp.hour * 60


    @property
    def time_series(self):
        if self.diff_with_source >= 0 and self.diff_with_source<= 2:
            return 0
        elif self.diff_with_source > 2 and self.diff_with_source <= 5:
            return 1
        elif self.diff_with_source > 5 and self.diff_with_source <= 10:
            return 2
        elif self.diff_with_source > 10 and self.diff_with_source <= 30:
            return 3
        elif self.diff_with_source > 30 and self.diff_with_source <= 60:
            return 4
        else:
            return -1



    def __str__(self):
        return f'{self.minute},{self.diff_with_source},{self.is_source_tweet},{self.is_rumor}'



def loadFile(event, filePath):
    # 1 -> rumors , 0 -> non-rumor
    type={1: "rumours",0:"non-rumours"}
    data=list()
    rumourCount = 0
    non_rumourCount=0
    #D:\BOOKS\twitterRumor\annotated-threads\charliehebdo\non-rumours
    rumor_folders=[f.replace("\\","/") for f in glob.glob(os.path.join(f'D:/BOOKS/twitterRumor/annotated-threads/{event}/{type[1]}',"*/"), recursive=False)]
    non_rumor_folders = [f.replace("\\","/") for f in glob.glob(os.path.join(f'D:/BOOKS/twitterRumor/annotated-threads//{event}/{type[0]}',"*/"), recursive=False)]

    '''    reading Rumor Folder  '''

    print(f"reading Rumor Folder of {event}")
    for count,f in enumerate(rumor_folders):
            rumourCount += 1

            source_tweet_path = f'{f}source-tweets/{os.path.basename(os.path.normpath(f))}.json'
            #print(source_tweet_path)
            source_tweet = json.load(open(source_tweet_path))
            source_obj = tweet(source_tweet["created_at"], 1, 1)
            vector = [0, 0, 0, 0, 0, 1]
            #data.append(source_obj)
            reactions_paths= [f.replace('\\','/') for f in glob.glob(f'{f}/reactions/*.json')]

            for l in reactions_paths:
                reaction_tweet = json.load(open(l))
                reaction_obj = tweet(reaction_tweet["created_at"], 0, 1,source_obj)
                #data.append(reaction_obj)
                if reaction_obj.time_series != -1 :
                    vector[reaction_obj.time_series] += 1

            data.append(vector)

            


    print(f"Finished reading Rumor Folder of {event}")


    '''  loading  non rumor folder  '''


    print(f"reading NON - Rumor Folder of {event}")

    for count, f in enumerate(non_rumor_folders):
        non_rumourCount+=1
        source_tweet_path = f'{f}source-tweets/{os.path.basename(os.path.normpath(f))}.json'
        #print(source_tweet_path)
        source_tweet = json.load(open(source_tweet_path))
        source_obj = tweet(source_tweet["created_at"],1, 0)
        #data.append(source_obj)
        reactions_paths = [f.replace('\\', '/') for f in glob.glob(f'{f}/reactions/*.json')]
        vector = [0, 0, 0, 0, 0, 0]

        for l in reactions_paths:
            reaction_tweet = json.load(open(l))
            reaction_obj = tweet(reaction_tweet["created_at"], 0, 0,source_obj)
            #data.append(reaction_obj)
            if reaction_obj.time_series != -1:
                vector[reaction_obj.time_series] += 1

        data.append(vector)
        #print('working...')

    print(f"Finished reading Rumor Folder of {event}")



    '''   writting data to csv file    '''

    print(f'Writting "{event} " data to csv file')
    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)
    with open(filePath, 'w', newline='') as file:
        writer = csv.writer(file)
        for d in data:
            writer.writerow(d)
            #print(d)


    print(f'SUMMARY = {event} has rumors= {rumourCount} and non-rumors={non_rumourCount} ')

    # return data description of loaded data
    return [event,rumourCount,non_rumourCount,rumourCount+non_rumourCount]










if __name__ == '__main__':
    events = ["charliehebdo", "ferguson", "germanwings-crash", "gurlitt", "ottawashooting", "putinmissing",
              "sydneysiege"]


    for e in events:
        data = loadFile(e, f'D:/BOOKS/5th Sem/AI Lab/AI-PROJECT/CSV_Files/{e}.csv')

        with open(os.path.join('D:/BOOKS/5th Sem/AI Lab/AI-PROJECT/CSV_Files/DataDescription.csv'), 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)

    '''
    data = pd.read_csv('D:/BOOKS/5th Sem/AI Lab/AI-PROJECT/CSV_Files/charliehebdo.csv')
    df = pd.DataFrame(data)
    #df.head(20)
    print(df.values)
    #plt.plot(df)
    #plt.show()
    '''
