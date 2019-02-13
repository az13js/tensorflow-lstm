# -*- coding: UTF-8 -*-
import random

def getVectorLength():
    table = " ;:'\",.?!`~#@%$^&*+=-|()[]}{\\/<>abcdefghijklmnopqrstuvwxyz0123456789"
    return len(table)

def getRandomChar():
    table = " ;:'\",.?!`~#@%$^&*+=-|()[]}{\\/<>abcdefghijklmnopqrstuvwxyz0123456789"
    return table[int(random.random() * (1 + len(table)))]

def charToList(ch):
    table = " ;:'\",.?!`~#@%$^&*+=-|()[]}{\\/<>abcdefghijklmnopqrstuvwxyz0123456789"
    try:
        key = table.index(ch.lower())
    except ValueError:
        key = 0
    res = []
    for i in range(len(table)):
        if i == key:
            res.append(1)
        else:
            res.append(0)
    return res

def listToChar(input_list):

    def getKeyOfMax(input_list):
        key = 0
        max_value = input_list[key]
        for i in range(len(input_list)):
            if input_list[i] > max_value:
                max_value = input_list[i]
                key = i
        return key

    table = " ;:'\",.?!`~#@%$^&*+=-|()[]}{\\/<>abcdefghijklmnopqrstuvwxyz0123456789"
    return table[getKeyOfMax(input_list)]

def readContext(file_name):
    f = open(file_name)
    s = f.read()
    f.close()
    return s

def getListSequenceFromFile(file_name):
    s = readContext(file_name)
    res = []
    for i in range(len(s)):
        res.append(charToList(s[i]))
    return res
