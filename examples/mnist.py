import torch,torchvision
import torchvision.transforms as transforms
import torch.nn as nn

print('start!')
# device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# super parameters
input_num=28*28
hidden_num=400
classes_num=10
learning_rate=0.001
batch_num=100
epoch_num=4
# data
train_dataset=torchvision.datasets.MNIST('../data',True,transform=transforms.ToTensor(),download=True)
test_dataset=torchvision.datasets.MNIST('../data',False,transform=transforms.ToTensor())
# dataloader
train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_num,shuffle=True)
test_dataloader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_num,shuffle=True)

# model
class NenuralModel(nn.Module):
    def __init__(self,input_num,hidden_num,classes_num):
        super(NenuralModel,self).__init__()
        self.fc1=nn.Linear(input_num,hidden_num)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(hidden_num,classes_num)
    def forward(self,x):
        out=self.fc1(x)
        out=self.relu(out)
        out=self.fc2(out)
        return out

model=NenuralModel(input_num,hidden_num,classes_num)
# loss and optimizer
cri=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
# run training
total=len(train_dataloader)
for epoch in range(epoch_num):
    for i,(images,labels) in enumerate(train_dataloader):
        images=images.reshape(-1,28*28).to(device)
        labels=labels.to(device)

        output=model(images)
        loss=cri(output,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if (i+1)%100==0:
            print('epoch [%s]/[%s], period [%s]/[%s], loss: %2.3f'%(epoch+1,epoch_num,i+1,total,loss.item()))


# run test
with torch.no_grad():
    correct=0
    total=0
    for images,labels in test_dataloader:
        images=images.reshape(-1,28*28).to(device)
        labels=labels.to(device)

        result=model(images)

        _,pred=torch.max(result.data,1)
        total+=labels.size(0)

        correct+=(pred==labels).sum().item()

    print('Accuracy of {} items:{} %'.format(total,100*correct/total))
# save
torch.save(model.state_dict,'mnist.ckpt')