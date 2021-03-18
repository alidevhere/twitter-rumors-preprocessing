import os
import glob
import json
import datetime
import csv
import random


class tweet:
    ''' Constructor for tweet        '''

    def __init__(self, timeDiff, freq , is_rumor):
        self.timeDiff = timeDiff
        self.is_rumor = is_rumor
        self.freq = freq

    @property
    def inc(self):
        self.freq += 1



def proper_round(num, dec=0):
    num = str(num)[:str(num).index('.') + dec + 2]
    if num[-1] >= '5':
        a = num[:-2 - (not dec)]  # integer part
        b = int(num[-2 - (not dec)]) + 1  # decimal part
        return float(a) + b ** (-dec + 1) if a and b == 10 else float(a + str(b))
    return float(num[:-1])



def loadFile(event, filePath):
    # 1 -> rumors , 0 -> non-rumor

    rumors_list = [tweet(timeDiff=0 ,freq= 0 , is_rumor=1 )]
    non_rumors_list = [tweet(timeDiff=0 ,freq= 0 , is_rumor=0 )]

    type = {1: "rumours", 0: "non-rumours"}
    data = list()
    rumourCount = 0
    non_rumourCount = 0


    # D:\BOOKS\twitterRumor\annotated-threads\charliehebdo\non-rumours
    rumor_folders = [f.replace("\\", "/") for f in
                     glob.glob(os.path.join(f'D:/BOOKS/twitterRumor/annotated-threads/{event}/{type[1]}', "*/"),
                               recursive=False)]
    non_rumor_folders = [f.replace("\\", "/") for f in
                         glob.glob(os.path.join(f'D:/BOOKS/twitterRumor/annotated-threads//{event}/{type[0]}', "*/"),
                                   recursive=False)]

    '''    reading Rumor Folder  '''

    print(f"reading Rumor Folder of {event}")
    for count, f in enumerate(rumor_folders):
        rumourCount += 1
        source_tweet_path = f'{f}source-tweets/{os.path.basename(os.path.normpath(f))}.json'
        reactions_paths = [f.replace('\\', '/') for f in glob.glob(f'{f}/reactions/*.json')]

        # Converting source tweet time stamp tos seconds
        source_tweet = json.load(open(source_tweet_path))
        source_time = datetime.datetime.strptime(source_tweet["created_at"], '%a %b %d %H:%M:%S %z %Y')
        source_time = source_time.hour*3600 + source_time.minute*60 + source_time.second


        for l in reactions_paths:
            reaction_tweet = json.load(open(l))
            reaction_time = datetime.datetime.strptime(reaction_tweet["created_at"], '%a %b %d %H:%M:%S %z %Y')
            reaction_time = reaction_time.hour * 3600 + reaction_time.minute * 60 + reaction_time.second
            time_Diff = reaction_time-source_time
            if time_Diff < 0:
                print(reaction_tweet["created_at"],' -  ',source_tweet['created_at'],'  = ',time_Diff)

                break

            break
            done = False
            while done != True:
                if int(time_Diff) < len(rumors_list):
                    #print('index=',int(time_Diff))
                    rumors_list[int(time_Diff)].inc
                    #done = True
                    break
                else:
                    rumors_list.append(tweet(timeDiff= len(rumors_list),freq= 0 , is_rumor=1 ))

    print(f"Finished reading Rumor Folder of {event}")

    '''  loading  non rumor folder  '''

    print(f"reading NON - Rumor Folder of {event}")
    '''
    for count, f in enumerate(non_rumor_folders):
        non_rumourCount += 1
        source_tweet_path = f'{f}source-tweets/{os.path.basename(os.path.normpath(f))}.json'
        # print(source_tweet_path)
        source_tweet = json.load(open(source_tweet_path))
        source_obj = tweet(source_tweet["created_at"], 1, 0)
        # data.append(source_obj)
        reactions_paths = [f.replace('\\', '/') for f in glob.glob(f'{f}/reactions/*.json')]
        vector = [0, 0, 0, 0, 0, 0]
        is_valid_source_tweet = False

        for l in reactions_paths:
            reaction_tweet = json.load(open(l))
            reaction_obj = tweet(reaction_tweet["created_at"], 0, 0, source_obj)
            # data.append(reaction_obj)
            if reaction_obj.time_series != -1:
                vector[reaction_obj.time_series] += 1
                is_valid_source_tweet = True

        if is_valid_source_tweet:
            data.append(vector)
        # print('working...')

    print(f"Finished reading Rumor Folder of {event}")
    '''
    '''   writting data to csv file    '''

    print(f'Writting "{event} " data to csv file')
    #random.shuffle(data)
    data = rumors_list
    with open(filePath, 'w', newline='') as file:
        writer = csv.writer(file)
        for d in data:
            writer.writerow(d)
            # print(d)

    print(f'SUMMARY = {event} has rumors= {rumourCount} and non-rumors={non_rumourCount} ')

    # return data description of loaded data
    return [event, rumourCount, non_rumourCount, rumourCount + non_rumourCount]



if __name__ == '__main__':

    loadFile('charliehebdo','D:/BOOKS/5th Sem/AI Lab/AI-PROJECT/CSV_Files/charliehebdo')