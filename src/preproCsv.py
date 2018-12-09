import pandas
from sklearn.utils import shuffle

#Reading the csv file
df = pandas.read_csv("Data_Entry.csv")

#Remove unnecessary columns
df.drop(['OriginalImagePixelSpacing_x', 'y', 'OriginalImage_Width', 'Height'], axis = 1, inplace = True)

#----------- Preprocess age column -----------

#Delete rows where age is not in years
age_filter = df['Patient_Age'].str.contains('[M|D]')
df = df[~age_filter]

#Convert to integer
df.loc[:, 'Patient_Age'] = df['Patient_Age'].apply(lambda x: str(x)[:-1]).astype(int)

#Delete rows where age is unrealistic
df.drop(df[df['Patient_Age'] > 120].index, inplace=True)


#----------- Reduce categories -----------

#Delete rows where are more disease
filter = df['Finding_Labels'].str.contains("\|")
df = df[~filter]

#Get the 5000 rows where category is No Finding
df_temp = df
filter = df['Finding_Labels'].str.contains("No Finding")
df = df[filter]
df = shuffle(df)
df = df.head(5000)
df['output'] = 0

#Get 5000 rows where category is not Not Finding
df_temp = df_temp.drop(df_temp[df_temp['Finding_Labels'] == "No Finding"].index)
df_temp = shuffle(df_temp)
df_temp = df_temp.head(5000)
df_temp['output'] = 1

#Concatenate the two dataframe
frames = [df, df_temp]
dataFrame = pandas.concat(frames)

dataFrame.drop(['Finding_Labels'], axis = 1, inplace = True)

#Shuffle the rows randomly
dataFrame = shuffle(dataFrame)
dataFrame['positive'] = 0
dataFrame['negative'] = 0

#Make one-hot encoding
for index, row in dataFrame.iterrows():
    dataFrame.loc[df['output'] == 1, 'positive'] = 1
    dataFrame.loc[df['output'] == 0, 'negative'] = 1

#Drop unused columns
dataFrame.drop(['output'], axis = 1, inplace = True)
dataFrame.drop(['Unnamed: 0'], axis = 1, inplace = True)

#Extract dataframe to a csv
dataFrame.to_csv('entry_data_edited.csv')
#Just for printing out the dataframe
pandas.set_option("max_columns", None)

#Create and extract train, validation and test dataset
train = dataFrame.head(5001)
train.to_csv('train.csv')

test = dataFrame.tail(2999)
test.to_csv('test.csv')

valid = pandas.concat([df, train, test]).drop_duplicates(keep=False)
valid.to_csv('valid.csv')
