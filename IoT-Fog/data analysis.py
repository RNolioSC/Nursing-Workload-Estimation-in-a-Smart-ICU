import mysql.connector
import argparse
import mariadb
from datetime import datetime, timedelta
from matplotlib import pyplot as plt

if __name__ == "__main__":
	try:
		conn = mariadb.connect(
		user="admin",
		password="admin",
		host="localhost",
		port=3306,
		database="bancodedados"
		)

	except mariadb.Error as e:
		print(e)
		sys.exit(1)
                
	cur = conn.cursor()
	try:
		cur.execute("select * from resultados")
		atendimentos = cur.fetchall()
	except mariadb.Error as e:
		print(e)

	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--dia", type=int)
	args = parser.parse_args()
	dia=args.dia
	
	data = [i[1] for i in atendimentos if i[1].day==dia]
	if not data:
		print("nenhum dado retornado")
		exit()
		
	data_by_profEnf = {}
	
	for i in atendimentos:
		if  i[1].day!=dia:
			continue
		profEnf = i[2]
		timestamp = i[1]
		if profEnf not in data_by_profEnf:
			data_by_profEnf[profEnf] = [timestamp]
		else:
			temp = data_by_profEnf.get(profEnf)
			temp.append(timestamp)
			data_by_profEnf[profEnf] = temp
		
			
	#data_enf = [[i[1], i[2]] for i in atendimentos if i[1].day==29]
	#list_profEnfs = []
	#for i in data_enf:
	#	if i[1] not in list_profEnfs:
	#		list_profEnfs.append(i[1])
	
	#print(data)
	#exit()
	#plt.plot([i[1] for i in atendimentos])
	
	#plt.eventplot(data, orientation='horizontal', colors='b')
	#plt.hlines(1, min(data), max(data), colors='k')
	#plt.show()
	#diffs=[data[0]-data[0]]
	#for i in range(1, len(data)):
	#	diffs.append(data[i]-data[i-1])
	#print(diffs)
	
	fig, axs = plt.subplots()
	#axs.eventplot(data, orientation='horizontal')
	#axs.hlines(0, min(data), max(data), colors='k')
	
	y_labels=[]
	pos_y=0
	all_pos_y=[]
	for profEnf in data_by_profEnf:
		axs.hlines(pos_y, min(data), max(data), colors='k')
		axs.eventplot(data_by_profEnf[profEnf], orientation='horizontal', lineoffsets=pos_y, linelengths=0.5)
		y_labels.append(profEnf)
		all_pos_y.append(pos_y)
		pos_y+=1
		
	axs.set_yticks(all_pos_y, y_labels, rotation=45)
	
	xticks = []
	for i in data:
		xticks.append(i.strftime("%H:%M")) # "%H:%M:%S"
	xticks2=[data[0].strftime("%H:%M:%S")]
	xticks2ref = [data[0]]
	for i in range(1, len(data)):
		if data[i]-data[i-1] > timedelta(minutes=30):
			xticks2.append(data[i].strftime("%H:%M:%S"))
			xticks2ref.append(data[i])
			
	if data[-1]- xticks2ref[-1] > timedelta(minutes=30):
		# add ultimo se necessario
		xticks2.append(data[-1].strftime("%H:%M:%S"))
		xticks2ref.append(data[-1])
				
	#axs.set_xticklabels(diffs)
	axs.set_xticks(xticks2ref, xticks2, rotation=45)
	#axs.set_yticks([])
	axs.set_title("Acessos no dia "+data[0].strftime("%d/%m"))
	plt.show()
		
