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


#Make one-hot encoding
df = pandas.get_dummies(df, columns=["Finding_Labels"])

#Shuffle the rows randomly
df = shuffle(df)


#Extract dataframe to a csv
df.to_csv('entry_data_edited.csv')
#Just for printing out the dataframe
pandas.set_option("max_columns", None)
print(df)