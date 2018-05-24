from random import shuffle

f = open('/home/yaserkl/wapo/data/cnn_dm/cnn/cnn.txt')
cnn = [line.strip() for line in f]
f.close()
f = open('/home/yaserkl/wapo/data/cnn_dm/dailymail/dailymail.txt')
dailymail = [line.strip() for line in f]
f.close()

cnn.extend(dailymail)
shuffle(cnn)

train_size = int(len(cnn)*0.92)
eval_size = int(len(cnn)*0.04)
test_size = len(cnn)-(train_size+eval_size)

train = cnn[0:train_size]
eval = cnn[train_size:train_size+eval_size]
test = cnn[train_size+eval_size:]

fw = open('/home/yaserkl/wapo/data/cnn_dm/train.txt','w')
for k in train:
    fw.write('{}\n'.format(k))
fw.close()

fw = open('/home/yaserkl/wapo/data/cnn_dm/eval.txt','w')
for k in eval:
    fw.write('{}\n'.format(k))
fw.close()

fw = open('/home/yaserkl/wapo/data/cnn_dm/test.txt','w')
for k in test:
    fw.write('{}\n'.format(k))
fw.close()

