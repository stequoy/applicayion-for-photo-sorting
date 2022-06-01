import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
import time
import os
import copy
import typer
from tqdm import tqdm
import shutil
from PIL import Image


app = typer.Typer()


# def imshow(inp, title):
#     mean = torch.tensor([0.485, 0.456, 0.406])
#     std = torch.tensor([0.229, 0.224, 0.225])
#     inp = inp.numpy().transpose((1, 2, 0))
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     plt.title(title)
#     plt.show()


@app.command(short_help="1st argument - directory with 2 sub directories \"training\" and \"evaluation\" which have n \
                        subdirectories, where n is the number of classes")
def train(root_directory: str, num_of_epochs: int = 4):

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    common_data = {}

    data_transforms = {
        'training': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'evaluation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    image_datasets = {
         'training': datasets.ImageFolder(os.path.join(root_directory, 'training'), data_transforms['training']),
         'evaluation': datasets.ImageFolder(os.path.join(root_directory, 'evaluation'), data_transforms['evaluation'])}

    data_loaders = {
         'training': torch.utils.data.DataLoader(image_datasets['training'], batch_size=4, shuffle=True),
         'evaluation': torch.utils.data.DataLoader(image_datasets['evaluation'], batch_size=4, shuffle=True)}

    dataset_sizes = {'training': len(image_datasets['training']),
                     'evaluation': len(image_datasets['evaluation'])}

    class_names = image_datasets['training'].classes

    common_data["classes"] = class_names

    with open('common_data.pickle', 'wb') as f:
        pickle.dump(common_data, f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(class_names)

    # Get a batch of training data
    inputs1, classes = next(iter(data_loaders['training']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs1)

    # imshow(out, title=[class_names[x] for x in classes])

    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        starting_time = time.time()

        best_model = copy.deepcopy(model.state_dict())  # download the best weights from previous iter.
        best_accuracy = 0.0  # initialize accuracy

        # calculating the loss and optimizing
        for epoch in range(num_epochs):  # calculating
            print('Epoch ', epoch, 'out of ', num_epochs - 1)
            print('----------------')
            for phase in ['training', 'evaluation']:
                if phase == 'training':
                    model.train()  # turn on the train mode
                else:
                    model.eval()   # turn on the evaluation mode

                epoch_cumulative_loss = 0.0
                epoch_cumulative_corrects = 0

                # iteration on data
                for inputs, labels in tqdm(data_loaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # if training, save the predictions and optimize
                    with torch.set_grad_enabled(phase == 'training'):
                        outputs = model(inputs)
                        _, predictions = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # optimize
                        if phase == 'training':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    # sum up the stats
                    epoch_cumulative_loss += loss.item() * inputs.size(0)
                    epoch_cumulative_corrects += torch.sum(predictions == labels.data)

                if phase == 'training':
                    scheduler.step()  # adjust the learning rate using scheduler

                epoch_loss = epoch_cumulative_loss / dataset_sizes[phase]
                epoch_accuracy = epoch_cumulative_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_accuracy))

                # copy the model
                if phase == 'evaluation' and epoch_accuracy > best_accuracy:
                    best_accuracy = epoch_accuracy
                    best_model = copy.deepcopy(model.state_dict())  # save the best model
            print()

        time_passed = time.time() - starting_time
        print('Training time: {:.0f}m {:.0f}s'.format(time_passed // 60, time_passed % 60))
        print('Best evaluation Acc.: {:4f}'.format(best_accuracy))
        model.load_state_dict(best_model)  # save the best model
        return model

    # declaration of the model
    model = models.resnet152(pretrained=True)

    # fix the parameters in the model, so we don't have to optimize them
    for param in model.parameters():
        param.requires_grad = False

    # params of new layers have grad == True
    num_of_features = model.fc.in_features
    # add new linear layer to the end with 5 outputs
    # in this example there is only 5 classes (custom groups)
    model.fc = nn.Linear(num_of_features, 5)

    model = model.to(device)  # pass to GPU if possible

    criterion = nn.CrossEntropyLoss()  # criterion for optimization = cross entropy loss

    # set the optimization method to be stochastic gradient descent
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

    # multiply learning rate by 0.1 every 7 epoch to achieve greater precision
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # train model for one epoch (just an example the number of epochs of course should be greater)
    model = train_model(model, criterion, optimizer, lr_scheduler, num_epochs=num_of_epochs)

    torch.save(model.state_dict(), "model.pth")


@app.command(short_help="classify images; 1st argument - path to \
                         the source directory 2nd argument target directory")
def classify(source, target):

    with open('common_data.pickle', 'rb') as file:
        common_data = pickle.load(file)

    model = models.resnet152()
    model.fc = torch.nn.Linear(model.fc.in_features, len(common_data["classes"]))
    model.load_state_dict(torch.load("model.pth"))

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    transformation = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def predict(photo):
        return torch.argmax(model(photo.reshape(1, 3, 224, 224)))

    target_categories_paths = []

    for i in common_data["classes"]:
        p = os.path.join(target, i)
        target_categories_paths.append(p)
        if not os.path.exists(p):
            os.mkdir(p)
    for f in os.listdir(source):
        if not f.startswith('.'):
            path_to_the_photo = os.path.join(source, f)
            orig = Image.open(path_to_the_photo)
            transformed_image = transformation(orig)
            prediction = predict(transformed_image)
            shutil.move(os.path.join(source, f), target_categories_paths[prediction])


if __name__ == "__main__":
    app()
