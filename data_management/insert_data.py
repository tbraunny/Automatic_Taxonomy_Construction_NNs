import pandas as pd
import psycopg2

####################################################
# Change fields as needed throughout the project

# Connect to database
def connectToDatabase(): 
    connection = psycopg2.connect(dbname= "nn_papers_dataset" , user= "atc_nn" , password= "7%vE3yr2")
    cursor = connection.cursor()

    return connection , cursor

# Properly disconnect from server
def closeDatabase(connection , cursor):
    connection.commit()
    cursor.close()
    connection.close()

conn , cur = connectToDatabase()
df = pd.read_csv("dataset_papers.csv")

paper_table = df[['title' , 'author' , 'year']]
class_table = df[['model_type' , 'algorithm' , 'activation' , 'datatype' , 'layers' , 'parameters' , 'task']]

for i , row in paper_table.iterrows():
    cur.execute(f"""INSERT INTO paper_info (title , author , year) VALUES (%s , %s , %s)""" , 
                (row['title'] , row['author'] , row['year']))

for i , row , in class_table.iterrows():
    cur.execute(f"""INSERT INTO classification (model_type , algorithm , activation , datatype , layers , parameters , task) 
                VALUES (%s , %s , %s , %s , %s , %s , %s)""" , 
               (row['model_type'] , row['algorithm'] , row['activation'] , row['datatype'] ,
                row['layers'] , row['parameters'] , row['task']))
    

print("Queries executed")

closeDatabase(conn , cur)