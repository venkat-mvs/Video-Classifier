
from matplotlib import pyplot as plt
import numpy as np

lstm_x = np.load('lstm_x.npy')
lstm_y = np.load('lstm_y.npy')

print(lstm_y[0])

classes = ['SkateBoarding','Running','RidingHorse','GolfSwinging']
from sklearn.preprocessing  import LabelEncoder

lb = LabelEncoder().fit(classes)

print(len(lstm_x))
i = int(input("Enter i : "))
label = lb.inverse_transform([np.argmax(lstm_y[i])])[0]
print(f"{label} class as :")

def show(x):
	w = h = 10 
	fig = plt.figure(figsize = (16,32))
	columns = 5
	rows = 2
	for i in range(1,columns*rows +1):
		img = x[i-1]
		img.resize((16,32))
		fig.add_subplot(rows,columns,i)
		plt.imshow(img,cmap='Dark2')
	plt.show()
show(lstm_x[i])
