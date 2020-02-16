import os
import torch.utils.data
from self_spiking_model import *
from snn_dataset import SNNDataset


data_path = './dataset'
names = 'spiking_model'
size = 100
preload = True
loadckpt = True
total = 0
correct = 0
sum_acc = 0
test_times = 20
batch_size = 10

snn = SCNN()
snn.to(device)
path = './checkpoint/ckptspiking_model.t7'
optimizer = torch.optim.Adam(snn.parameters(), lr=0.01)
checkpoint = torch.load(path)
snn.load_state_dict(checkpoint['net'])
acc = checkpoint['acc']
acc_record = checkpoint['acc_record']
epoch = checkpoint['epoch']

test_dataset = SNNDataset(data_path, size=size, train=True, preload=preload)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

for i in range(test_times):
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = snn(inputs)
        labels_ = torch.zeros(batch_size, 11).scatter_(1, targets.view(-1, 1) - 1, 1)
        _, predicted = outputs.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets-1).sum().item())
        acc = 100. * float(correct) / float(total)
        sum_acc += acc
        if (batch_idx % 5 == 0):
            print((predicted+1).tolist())
            print(targets.tolist())
            print( ' Acc: %.5f' % acc)
    print( '[Test %d] The Acc on %d test examples is %.5f \n\n' %(i+1,batch_size,sum_acc/((i+1)*batch_size)))