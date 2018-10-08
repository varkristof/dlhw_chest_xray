import pandas

#Reading the csv file
entry_data = pandas.read_csv("Data_Entry.csv")

#Delete rows where are more disease
patternDel = "\|"
filter = entry_data['Finding_Labels'].str.contains(patternDel)
df = entry_data[~filter]

#Just for printing out the dataframe
pandas.set_option("max_columns", 4)
print(df)