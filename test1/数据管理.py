# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 19:30:12 2022

@author: webot
"""
import pandas as pd
import pymysql

#然后，使用pandas的read_excel函数读取Excel文件的数据：

df = pd.read_csv('dataStcok/000001.csv',encoding=('gbk'))

#接下来，使用pymysql的connect函数连接到mysql服务器，并使用cursor函数获取游标：

conn = pymysql.connect(host='localhost', user='root', password='webots', database='test1')
cursor = conn.cursor()
# 在test1 数据库里面创建名字为test的表格 有日期和股票代码
#cursor.execute("CREATE TABLE test (id INT NOT NULL PRIMARY KEY AUTO_INCREMENT,日期 DATE NOT NULL,股票代码 VARCHAR(255) NOT NULL,名称 VARCHAR(255) NOT NULL);")
# 在test1数据库里面的test表格添加一列
#cursor.execute("ALTER TABLE test ADD COLUMN 收盘价 DECIMAL(10, 2) NOT NULL;")

# 在test1数据库里面创建名字为test的表格修改其中一列

#cursor.execute("ALTER TABLE test CHANGE COLUMN 收盘价 价格 DECIMAL(10, 2) NOT NULL;")

cursor.execute("ALTER TABLE test CHANGE COLUMN 收盘价 收盘价 DECIMAL(10, 3) NOT NULL;")
#然后，使用pandas的iterrows函数遍历df中的每一行，并使用pymysql的execute函数执行SQL语句：
#%%
for index, row in df.iterrows():
    
    
    print(index,row)
    
    sql = "INSERT INTO test(日期, 股票代码, 名称) VALUES (%s, %s, %s)"
    cursor.execute(sql, (row['日期'], row['股票代码'], row['名称']))
    
#%% 最后，使用conn的commit函数提交事务，并关闭连接：
conn.commit()
conn.close()
