# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 05:11:50 2022

@author: webot
"""
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
import re
import os
a=os.getcwd()+'\dataStcok'

options = Options()
options.add_experimental_option("prefs", {
    "download.default_directory": a,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True
})

driver=webdriver.Edge(options=options)


def getStockCode(num):
    xp='//*[@id="contents"]'
    url=f'https://www.360docs.net/doc/303337145-{num}.html'
    driver.get(url)
    text=driver.find_element(By.XPATH,xp).text
    text0=re.findall(r'\d+',text)
    #driver.quit()
    return (text,text0)
   
    
    





def getStockData(num):
    xp0='//*[@id="downloadData"]'
    xp1='//*[@id="dropBox1"]/div[2]/form/table[2]/tbody/tr[1]/td[2]/input[3]'
    xp2='//*[@id="dropBox1"]/div[2]/form/table[2]/tbody/tr[2]/td[2]/input[3]'
    xp3='//*[@id="dropBox1"]/div[2]/form/div[3]/a[1]'
    url=f'http://quotes.money.163.com/trade/lsjysj_{num}.html'
    driver.get(url)
    driver.find_element(By.XPATH,xp0).click()
    driver.find_element(By.XPATH,xp1).click()
    driver.find_element(By.XPATH,xp2).click()
    driver.find_element(By.XPATH,xp3).click()
for i in range(1,27):
    a,b=getStockCode(i)
    for j in b:
        getStockData(j)

