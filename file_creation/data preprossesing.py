import csv
import json
import pandas as pd




def read_csv(filename):
    with open(filename, encoding='MacRoman', newline='') as csvfile:
        csvdata = csv.DictReader(csvfile)
        rows = [row for row in csvdata]
    return rows

def write_json(data, filename):
    with open(filename, 'w') as jsonfile:
        json.dump(data, jsonfile)

def read_user_files(user_filename):
    with open(user_filename, 'r') as user_file:
        user_lines = [line.strip() for line in user_file.readlines()]

    data = []

    for i, line in enumerate(user_lines):
        articles = line.split(' ')
        del articles[0]
        for article in articles:
            data.append([i, article, 1.0])

    users_df = pd.DataFrame(data, columns=['user', 'article', 'rating'])
    users_df['article'] = users_df['article'].astype(int)

    return users_df

if __name__ == '__main__':
    data_row_csv = 'C:\\Users\\Matil\\Código\\data\\citeulike-a\\data_row.csv' #ler o arquivo csv
    data_row_json = 'C:\\Users\\Matil\\Código\\data\\file_creation\\articles.json' #Arquivo JSON resultante da conversão do arquivo data_row.csv

    rows = read_csv(data_row_csv)
    write_json(rows, data_row_json)

    users_file = 'C:\\Users\\Matil\\Código\\data\\citeulike-a\\users.txt'

    users_df = read_user_files(users_file)

    users_json = 'C:\\Users\\Matil\\Código\\data\\file_creation\\users.json'
    write_json(users_df.to_dict('records'), users_json)

