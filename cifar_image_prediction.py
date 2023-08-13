#import modules i prefer to use torch instead tensorflow 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F
from tkinter import *
from PIL import Image as PILImage, ImageTk

from tkinter import filedialog

#program shows if you use CPU or GPU 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    print("Using GPU ")
else:
    print("Using CPU Slower")


# Data processing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

# Neural Network  Class create Network 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Training loop #10 epoch may not enough but fast
for epoch in range(10):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}")
            running_loss = 0.0

print('Finished Training')
torch.save(net.state_dict(), 'cifar10_model.pth')

# Load trained model for inference
model = Net().to(device)
model.load_state_dict(torch.load('cifar10_model.pth', map_location=device))
model.eval()

#CIFAR 10 class labels 
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#given an image 
def predict_image(img):
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        predicted_class_idx = output.argmax(dim=1).item()
    return class_labels[predicted_class_idx]

#not necessary but user correction with new window 
def get_user_correction(predicted_label):
    correction_window = Toplevel(root)
    correction_window.title("Correct the Prediction")

    Label(correction_window, text=f"Predicted: {predicted_label}. If incorrect, please select the correct label.").pack(pady=10)

    # Create a dropdown menu with class labels
    correct_label_var = StringVar(correction_window)
    correct_label_var.set(predicted_label)  # set the default value to the predicted label
    dropdown = OptionMenu(correction_window, correct_label_var, *class_labels)
    dropdown.pack(pady=10)

    def on_ok():
        correction_window.selected_label = correct_label_var.get()
        correction_window.destroy()

    Button(correction_window, text="OK", command=on_ok).pack(pady=10)
    
    correction_window.mainloop()
    return class_labels.index(correction_window.selected_label)  # Return the index of the selected label



def open_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    img = PILImage.open(file_path)
    
    img_for_prediction = img.resize((32, 32))
    predicted_label = predict_image(img_for_prediction)
    label.config(text=f"Prediction: {predicted_label}")
    img_for_display = img.resize((300, 300))
    tk_image = ImageTk.PhotoImage(img_for_display)
    canvas.config(width=300, height=300)
    canvas.image = tk_image
    canvas.create_image(0, 0, anchor=NW, image=tk_image)
    
    # Assuming you have a function to get corrected label from user
    #corrected_label_idx = get_user_correction(predicted_label) if predicted_label != expected_label else class_labels.index(predicted_label)
    corrected_label_idx = get_user_correction(predicted_label)


    if corrected_label_idx is not None:
        # Convert image to tensor
        img_tensor = transform(img_for_prediction).unsqueeze(0).to(device)
        label_tensor = torch.tensor([corrected_label_idx]).to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        outputs = net(img_tensor)
        loss = criterion(outputs, label_tensor)
        loss.backward()
        optimizer.step()

        # Optional: Save the updated model
        torch.save(net.state_dict(), 'cifar10_model.pth')



#simple GUI for select image and we decide what to do with our model 

root = Tk()
root.title("Image Classifier Using CIFAR-10")
root.geometry("600x500")
root.config(bg="lightgray")
canvas = Canvas(root, width=300, height=300, bg='white')

#canvas.pack(pady=20)
canvas.place(x=30,y=10)
label = Label(root, text="Prediction: None", font=("Arial", 12))
#label.pack(pady=20)
label.place(x=20,y=450)
button = Button(root, text="Open Image",command=open_image)
#btn.pack(pady=20)
button.place(x=50,y=400)
root.mainloop()


