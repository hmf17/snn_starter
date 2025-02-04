from __future__ import print_function
import os
import time
import torch.utils.data
from self_spiking_model import *
from snn_dataset import SNNDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
names = 'spiking_model'
data_path = './dataset'
preload = True
size_to_train = 5000
size_to_test = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using cuda" if torch.cuda.is_available() else "Using cpu")
print("batch_size = %d, thresh = %.2f " %(batch_size,thresh))
train_dataset = SNNDataset(data_path, size=size_to_train, train=True, preload=preload)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_dataset = SNNDataset(data_path, size=size_to_test, train=False, preload=preload)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])

snn = SCNN()
snn.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    running_loss = 0
    start_time = time.time()
    for i, (images, labels) in enumerate(train_loader):
        snn.zero_grad()
        optimizer.zero_grad()

        images = images.float().to(device)
        outputs = snn(images)
        labels_ = torch.zeros(batch_size, 11).scatter_(1, labels.long().view(-1, 1)-1, 1)
        loss = criterion(outputs.cpu(), labels_)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if (i+1)%25 == 0:
             print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                    %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size,running_loss ))
             running_loss = 0
             print('Time elasped:', time.time()-start_time)
    correct = 0
    total = 0
    optimizer = lr_scheduler(optimizer, epoch, learning_rate, 40)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = snn(inputs)
            labels_ = torch.zeros(batch_size, 11).scatter_(1, targets.view(-1, 1)-1, 1)
            loss = criterion(outputs.cpu(), labels_)
            _, predicted = outputs.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets-1).sum().item())
            if (batch_idx+1) % 25 ==0:
                acc = 100. * float(correct) / float(total)
                print(batch_idx, len(test_loader),' Acc: %.5f' % acc)

    print('Iters:', epoch+1)
    print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))
    acc = 100. * float(correct) / float(total)
    acc_record.append(acc)
    if epoch % 20== 0:
        print(acc)
        print('Saving..','\n\n\n')
        state = {
            'net': snn.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'acc_record': acc_record,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt' + names + '.t7')
        best_acc = acc