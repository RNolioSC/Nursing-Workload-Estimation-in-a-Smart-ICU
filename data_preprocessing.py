import csv

if __name__ == '__main__':
    with open("atendimentos_nn.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='|')
        next(reader, None)  # skip the headers
        tabela = [row for row in reader]
    #print(data_read)
