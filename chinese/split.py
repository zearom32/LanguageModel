n = 1485264

i1 = int(n*0.8)
i2 = int(n*0.9)
i = 0
f = open('data.txt','r')
f1 = open('ch.train.txt','w')
f2 = open('ch.valid.txt','w')
f3 = open('ch.test.txt','w')


for line in f:
    if i < i1:
        f1.write(line)
    elif i < i2:
        f2.write(line)
    else:
        f3.write(line)
    i = i + 1

f1.close()
f2.close()
f3.close()

