# -*- coding: utf-8 -*-
"""
Created on Thu May  2 17:19:36 2019

@author: Zibin Guan, Minjuan Zhang
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ast import literal_eval
import json
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Importing the dataset
dataset = pd.read_csv('train.csv')
dataset = dataset.drop(['id','belongs_to_collection','title','imdb_id','overview','original_title','poster_path'],axis=1)
#Data preprocessing
dataGenres = dataset.loc[:,'genres']
colGenresIDList=[]
dataHomepage = dataset.loc[:,'homepage']

dataOriLang = dataset.loc[:,'original_language']
colsOriLang=[]
dataProductionCompanies = dataset.loc[:,'production_companies']
colsProdComp=[]
dataProductionCountries = dataset.loc[:,'production_countries']
colsProdCountries=[]
dataReleaseDate = dataset.loc[:,'release_date']
dataSpokenLang = dataset.loc[:,'spoken_languages']
colsSpokenLang=[]
dataStatus = dataset.loc[:,'status']
colsStatus = []
dataTagline = dataset.loc[:,'tagline']
dataKeywords = dataset.loc[:,'Keywords']
colsKeywords=[]
dataCast = dataset.loc[:,'cast']
colCast=[]
dataCrew = dataset.loc[:,'crew']
colCrew=[]


releaseDate = dataset["release_date"].str.split("/", n = 2, expand = True) 

dataset["month"]= 'm'+ releaseDate[0]
dataset["day"]= 'd' + releaseDate[1]
dataset["year"]= 'y'+releaseDate[2]
dataMonth = dataset.loc[:,'month']
dataDay = dataset.loc[:,'day']
dataYear= dataset.loc[:,'year']
colMonth=[]
colDay=[]
colYear=[]


def dummyEncoderFunction(seriesData,listData):
    for i in range(len(dataset)):
        if pd.isnull(dataStatus[i]):
            pass
        else:
            if (seriesData.iloc[i] not in listData):
                listData.append(seriesData.iloc[i])
                dataset[seriesData.iloc[i]] = 0
                dataset.iloc[i,dataset.columns.get_loc(seriesData.iloc[i])] = 1
            else:
                dataset.iloc[i,dataset.columns.get_loc(seriesData.iloc[i])] = 1
    dataset.drop(dataset.columns[len(dataset.columns)-1], axis=1, inplace=True)
def dummyDictionary(seriesData,listData,idNum,idName): 
    for i in range(len(dataset)):
        if not pd.isnull(seriesData[i]):
            n = json.dumps(seriesData[i])
            temp_json = json.loads(n)
            python_dict = literal_eval(temp_json)
            for j in range(len(python_dict)):
                if (python_dict[j][idNum] not in listData):
                    listData.append(python_dict[j][idNum])
                    dataset[python_dict[j][idName]] = 0
                    dataset.iloc[i,dataset.columns.get_loc(python_dict[j][idName])] = 1
                else:
                    dataset.iloc[i,dataset.columns.get_loc(python_dict[j][idName])] = 1
        else:
            pass
    dataset.drop(dataset.columns[len(dataset.columns)-1], axis=1, inplace=True)

def YesOrNo(colName,seriesData):
    dataset[colName] = 0
    for i in range(len(dataset)):
        if pd.isnull(seriesData[i]):
            pass
        else:
            dataset.iloc[i,dataset.columns.get_loc(colName)] = 1

dataset = dataset.rename(index=str, columns={"popularity": "pupularityOrg"})  
#Genres
dummyDictionary(dataGenres,colGenresIDList,'id','name')

#homepage
YesOrNo('HasHomePage',dataHomepage)

#original langauge
dummyEncoderFunction(dataOriLang,colsOriLang)

#production company
dummyDictionary(dataProductionCompanies,colsProdComp,'id','name')

#production countries
dummyDictionary(dataProductionCountries,colsProdCountries,'iso_3166_1','name')

#spoken language
dummyDictionary(dataSpokenLang,colsSpokenLang,'iso_639_1','name')

#status langauge
dummyEncoderFunction(dataStatus,colsStatus)

#tagline
YesOrNo('HasTagLine',dataTagline)

#keywords
dummyDictionary(dataKeywords,colsKeywords,'id','name')

dummyEncoderFunction(dataMonth,colMonth)
dummyEncoderFunction(dataDay,colDay)
dummyEncoderFunction(dataYear,colYear)

'''
#cast
dummyDictionary(dataCast,colCast,'id','name')

#crew
dummyDictionary(dataCast,colCast,'id','name')
'''
#drop cols
dataset_1 = dataset.drop(['genres','homepage','original_language','production_companies','month','day','year','production_countries','release_date','spoken_languages','status','tagline','Keywords','cast','crew'],axis=1)
'''
# Check any number of columns with NaN
print(dataset_1.isnull().any().sum(), ' / ', len(dataset_1.columns))
# Check any number of data points with NaN
print(dataset_1.isnull().any(axis=1).sum(), ' / ', len(dataset_1))
'''
dataset_2 = dataset_1.dropna()

y = dataset_2.loc[:, 'revenue'].values
X = dataset_2.drop(['revenue'],axis=1).iloc[:, :].values


features = dataset_2.drop(['revenue'],axis=1).loc[:,:].columns.tolist()
target = dataset_2.loc[:,'revenue'].name
from scipy.stats import pearsonr
correlations = {}
for f in features:
    data_temp = dataset_2[[f,target]]
    x1 = data_temp[f].values
    x2 = data_temp[target].values
    key = f
    correlations[key] = pearsonr(x1,x2)[0]

data_correlations = pd.DataFrame(correlations, index=['Value']).T
top100 = data_correlations.loc[data_correlations['Value'].abs().sort_values(ascending=False).index].head(100)
top100['vs'] = top100.index


X = dataset_2[['budget','pupularityOrg','Adventure','superhero','marvel cinematic universe',
               'HasHomePage','3d','Walt Disney Pictures','orcs',
               'based on comic','Revolution Sun Studios','runtime','hobbit','middle-earth (tolkien)',
               'Hasbro Studios','shield','United States of America','WingNut Films','anthropomorphism'
               ,'middle-earth (tolkien)',
               'Hasbro Studios','shield','United States of America','WingNut Films','anthropomorphism','race','elves','wizard',
               'Action','duringcreditsstinger','batman','HasTagLine','superhero team',
               'imax','aftercreditsstinger','DC Entertainment','transformers','Pixar Animation Studios',
               'Jerry Bruckheimer Films','dwarves','Fantasy','Legendary Pictures','Family',
               'swashbuckler','Drama','broom','school of witchcraft','gotham city','dc comics','Syncopy',
               'sequel','Media Rights Capital (MRC)','Blue Sky Studios','saving the world','ice age','m6','super powers',
               'Di Bonaventura Pictures','One Race Films','riddle','en','sword and sorcery','giant robot',
               'Indochina Productions','Twentieth Century Fox Animation','east india trading company','criminal underworld',
               'catwoman','English','ring','New Zealand','Science Fiction','Heyday Films','Animation','crime fighter','speed',
               'tragic hero','secret identity','death star','Walt Disney Animation Studios','Hurwitz Creative', 
               'alice in wonderland','british secret service','rookie cop','vision','dc extended universe','goblin',
              'based on cartoon','exotic island',
               'scarecrow','district attorney','criminal mastermind']]
'''
,'middle-earth (tolkien)',
               'Hasbro Studios','shield','United States of America','WingNut Films','anthropomorphism',
               'Colorado Office of Film, Television & Media','Abu Dhabi Film Commission',
               'Québec Production Services T    ax Credit','muscle car','race','elves','wizard',
               'Action','duringcreditsstinger','batman','HasTagLine','superhero team',
               'imax','aftercreditsstinger','DC Entertainment','transformers','Pixar Animation Studios',
               'Jerry Bruckheimer Films','dwarves','Fantasy','Legendary Pictures','Family',
               'swashbuckler','Drama','broom','school of witchcraft','gotham city','dc comics','Syncopy',
               'sequel','Media Rights Capital (MRC)','Blue Sky Studios','saving the world','ice age'     
         '''      
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

'''
# Feature Scaling
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
'''


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000)
regressor.fit(X, y)

y_pred = regressor.predict(X_test)

#regressor.score(X_test,y_test)



'''import test file and clean the dataset then do the prediction'''




# Importing the test dataset
dataset = pd.read_csv('test.csv')
dataset = dataset.drop(['id','belongs_to_collection','title','imdb_id','overview','original_title','poster_path'],axis=1)
#Data preprocessing
dataGenres = dataset.loc[:,'genres']
colGenresIDList=[]
dataHomepage = dataset.loc[:,'homepage']

dataOriLang = dataset.loc[:,'original_language']
colsOriLang=[]
dataProductionCompanies = dataset.loc[:,'production_companies']
colsProdComp=[]
dataProductionCountries = dataset.loc[:,'production_countries']
colsProdCountries=[]
dataReleaseDate = dataset.loc[:,'release_date']
dataSpokenLang = dataset.loc[:,'spoken_languages']
colsSpokenLang=[]
dataStatus = dataset.loc[:,'status']
colsStatus = []
dataTagline = dataset.loc[:,'tagline']
dataKeywords = dataset.loc[:,'Keywords']
colsKeywords=[]
dataCast = dataset.loc[:,'cast']
colCast=[]
dataCrew = dataset.loc[:,'crew']
colCrew=[]
releaseDate = dataset["release_date"].str.split("/", n = 2, expand = True) 

dataset["month"]= 'm'+ releaseDate[0]
dataset["day"]= 'd' + releaseDate[1]
dataset["year"]= 'y'+releaseDate[2]
dataMonth = dataset.loc[:,'month']
dataDay = dataset.loc[:,'day']
dataYear= dataset.loc[:,'year']
colMonth=[]
colDay=[]
colYear=[]


dataset = dataset.rename(index=str, columns={"popularity": "pupularityOrg"})  
#Genres
dummyDictionary(dataGenres,colGenresIDList,'id','name')

#homepage
YesOrNo('HasHomePage',dataHomepage)

#original langauge
dummyEncoderFunction(dataOriLang,colsOriLang)

#production company
dummyDictionary(dataProductionCompanies,colsProdComp,'id','name')

#production countries
dummyDictionary(dataProductionCountries,colsProdCountries,'iso_3166_1','name')

#spoken language
dummyDictionary(dataSpokenLang,colsSpokenLang,'iso_639_1','name')

#status langauge
dummyEncoderFunction(dataStatus,colsStatus)

#tagline
YesOrNo('HasTagLine',dataTagline)

#keywords
dummyDictionary(dataKeywords,colsKeywords,'id','name')
dummyEncoderFunction(dataMonth,colMonth)
dummyEncoderFunction(dataDay,colDay)
dummyEncoderFunction(dataYear,colYear)

dataset_1 = dataset.drop(['genres','homepage','original_language','production_companies','production_countries','release_date','spoken_languages','status','tagline','Keywords','cast','crew'],axis=1)

dataset_1 = dataset_1.replace(np.nan, 0)

X = dataset_1[['budget','pupularityOrg','Adventure','superhero','marvel cinematic universe',
               'HasHomePage','3d','Walt Disney Pictures','orcs','marvel comic','Marvel Studios',
               'based on comic','Revolution Sun Studios','runtime','hobbit','middle-earth (tolkien)',
               'Hasbro Studios','shield','United States of America','WingNut Films','anthropomorphism'
               ,'middle-earth (tolkien)',
               'Hasbro Studios','shield','United States of America','WingNut Films','anthropomorphism','race','elves','wizard',
               'Action','duringcreditsstinger','batman','HasTagLine','superhero team',
               'imax','aftercreditsstinger','DC Entertainment','transformers','Pixar Animation Studios',
               'Jerry Bruckheimer Films','dwarves','Fantasy','Legendary Pictures','Family',
               'swashbuckler','Drama','broom','school of witchcraft','gotham city','dc comics','Syncopy',
               'sequel','Media Rights Capital (MRC)','Blue Sky Studios','saving the world','ice age','m6','super powers',
               'Di Bonaventura Pictures','One Race Films','riddle','en','sword and sorcery','giant robot',
               'Indochina Productions','Twentieth Century Fox Animation','east india trading company','criminal underworld',
               'catwoman','English','ring','New Zealand','Science Fiction','Heyday Films','Animation','crime fighter','speed',
               'tragic hero','secret identity','death star','Walt Disney Animation Studios','Hurwitz Creative', 
               'alice in wonderland','british secret service','rookie cop','vision','dc extended universe','goblin',
              'based on cartoon','exotic island',
               'scarecrow','district attorney','criminal mastermind']]

y_pred = regressor.predict(X)
result = pd.DataFrame(data=y_pred)
result['id'] = np.arange(len(result))+3001
submission = result.to_csv(r'D:\2018 NJIT\2019 Spring\R\Project2\submission.csv',index=False)

