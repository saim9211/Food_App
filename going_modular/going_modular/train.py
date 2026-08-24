
import os 
import torch 
import data_setup, engine, model_builder, utils
from torchvision import transforms

# set up training hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

#set up directories
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

#set up target device
device = "cuda" if torch.cuda.is_available() else "cpu"
# creat transformations
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
# create dataloader with help of data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                 test_dir=test_dir,
                                                                                 transform=data_transform,
                                                                                 batch_size=BATCH_SIZE)
#create model with help of model_builder.py
model=model_builder.TinnyGG(input_shape=3, hidden_units=HIDDEN_UNITS, output_shape=len(class_names)).to(device)
# set loss and optimizer
loss_fn=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
#strat training with help of engine.py
engine.train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=NUM_EPOCHS,
                device=device)

#daved teh model with help of utlis.py
utils.save_model(model=model,
                    target_dir="models",
                    model_name="06_going_modular_tingvgg_model.pth")