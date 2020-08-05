# pkg
from __future__ import print_function
import os
import time
import datetime
import torch.utils.data
from PWCSNet import PWCSNet, batch_size, thresh, time_windows
from dataloader import *
from Losses import *
from tensorboardX import SummaryWriter

# parameters
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"


names = 'DVS_Optical_Flow-'
data_path = '/home/CBICR/hmf/dataset/use'
preload = True
retrained_flag = False  # load the pretrained model
learning_rate = 0.0001
num_epochs = 200000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using cuda" if torch.cuda.is_available() else "Using cpu")
print("batch_size = %d, thresh = %.2f " %(batch_size,thresh))

# dataset
train_dataset = DVSFlowDataset(data_path, window=time_windows, train=True, preload=preload)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# test_dataset = DVSFlowDataset(data_path, window=time_windows, train=False, preload=False)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])

model = PWCSNet(training=True)
model = nn.DataParallel(model, device_ids=[0,1,2])
model.to(device)
criterion = Train_loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
writer = SummaryWriter()  # default ./run

# Dacay learning_rate
def lr_scheduler(optimizer, epoch):
    """Decay learning rate by half every lr_update_list point."""
    lr_update_list = [50000, 100000, 150000, 200000]
    if epoch in lr_update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5
    return optimizer

if retrained_flag:
    print("Using pretrained model")
    # input the direcory of pretrained model
    log_dir = 'pretrained_model/epoch_600/ckpt_DVS_Optical_Flow-2020-05-30-18-14-49.t7'
    checkpoint = torch.load(log_dir)
    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']

for epoch in range(num_epochs):
    running_loss = 0
    start_time = time.time()
    for i, (images, flow_gt) in enumerate(train_loader):
        model.zero_grad()
        images = images.float().to(device)
        flow_pred = model(images)
        train_loss = criterion(flow_pred, flow_gt)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        writer.add_scalar('Train Loss', train_loss.data, epoch * len(train_dataset) + i * batch_size)
        # change according to the input dataset
        if (i+1)%10 == 0:
            print ('Train Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                %(epoch+start_epoch+1, num_epochs, i+1, len(train_dataset)/batch_size, train_loss.item()))
            print('Time spent:', time.time()-start_time)

    optimizer = lr_scheduler(optimizer, epoch)
    writer.add_scalar('Train Spend Time', time.time()-start_time, epoch+1)

    # run test every epoch of train
    # test_EPE = 0
    # EPE_record = []
    # total = 0
    # with torch.no_grad():
    #     for batch_idx, (inputs, flow_gt) in enumerate(test_loader):
    #         inputs = inputs.float().to(device)
    #         optimizer.zero_grad()
    #         flow_pred = model(inputs)
    #         test_loss = criterion(flow_pred, flow_gt)
    #         # test_EPE = EPE(flow_gt, flow_pred[0])
    #         if (batch_idx+1) % 10 == 0:
    #             print(batch_idx, len(test_loader),' Loss: %.5f' % test_loss)
    #     writer.add_scalar('Test Loss', test_loss, epoch)
    #     # writer.add_scalar('Test EPE', test_EPE, epoch)

    # EPE_record.append(test_EPE)
    if epoch % 50 == 0:
        # print(EPE)
        print('Saving..', '\n\n\n')
        state = {
            'net': model.state_dict(),
            'epoch': epoch + start_epoch,
            'optimizer': optimizer.state_dict(),
            # 'EPE_record': EPE_record,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_' + names  + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.t7')
        torch.save(model.state_dict(), './checkpoint/model_' + names + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.t7')