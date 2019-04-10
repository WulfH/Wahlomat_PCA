import numpy as np
import json
import matplotlib.pyplot as plt

with open("party.json", "r") as read_file:
	rawpart = json.load(read_file)
part=[]
for item in rawpart:
	part.append(item["name"])

with open("opinion.json", "r") as read_file:
	rawdata = json.load(read_file)

data = np.zeros((32,38)) #party, statement

for item in rawdata:
	ans=item["answer"]
	if ans == 0:
		data[item["party"]][item["statement"]]=1
	else:
		data[item["party"]][item["statement"]]=ans-2

mean = np.mean(data, axis=0)
cov  = np.cov(data.T)
ndata= data-mean
eigVal, eigVec = np.linalg.eig(cov)

x,y = ndata[:,:].dot(eigVec[:,0]).real, ndata[:,:].dot(eigVec[:,1]).real

plt.scatter(x,y)
for i, txt in enumerate(part):
    plt.annotate(txt, (x[i], y[i]))
plt.show()
