


# 代码每日收盘后运行

import subprocess
import threading
import time

start_time = time.time()
# 程序代码

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
    
    
end_time = time.time()

total_time = end_time - start_time
minutes, seconds = divmod(total_time, 60)

if minutes>5:
    print('保存有效股票代码')
else:
    print('代码可能没有运行完，请重新运行')
#%%