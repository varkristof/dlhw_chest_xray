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

#Delete rows where categories are not in the four class 
df_temp = df
filter = df['Finding_Labels'].str.contains("Atelectasis|Infiltration|Pneumothorax|Edema|"
                                           "Emphysema|Fibrosis|Effusion|Pneumonia|Nodule|"
                                           "Mass|Hernia|No Finding")
df = df[~filter]

#Get 1465 random rows where category is No Finding
df_temp = df_temp.drop(df_temp[df_temp['Finding_Labels'] != "No Finding"].index)
df_temp = shuffle(df_temp)
df_temp = df_temp.head(1465)

#Concatenate the two dataframe
frames = [df, df_temp]
dataFrame = pandas.concat(frames)


#Make one-hot encoding
dataFrame = pandas.get_dummies(dataFrame, columns=["Finding_Labels"])

#Shuffle the rows randomly
dataFrame = shuffle(dataFrame)


#Extract dataframe to a csv
dataFrame.to_csv('entry_data_edited.csv')
#Just for printing out the dataframe
pandas.set_option("max_columns", None)
print(dataFrame)
