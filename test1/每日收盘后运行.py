import subprocess
import threading

#%%
# [0到13][0,1,(2,3,4,5,6,7,8,9,10,11),12,13]
c=[2,3,4,5,6,7,8,9,10,11]
def ma(a,b):
    subprocess.run(["python", "getMainCode.py",str(a)])
   
threads = []
for i in c:
    t = threading.Thread(target=ma, args=(i,0))
    threads.append(t)
    t.start()

# wait for all threads to finish
for t in threads:
    t.join()