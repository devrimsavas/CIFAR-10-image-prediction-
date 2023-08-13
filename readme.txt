Machine Learning : Using CIFAR 10 
summary: This application showcases the blend of machine learning and GUI programming to provide a real-time interactive experience.
Please read carefully about program: Because program takes long time to run tki interface... 
frankly, there is not any very basic way to use such machine learing. Because on the contrary of many people claim , modules do most job not programmer. .  In other worlds, to talk about machine learning does not make you clever. :D we just take module adjust and use. That is it. 
the main problem with such modelling is not to create model but what to do with this model: So we just create an interface using this model. 
the model i wrote is very standard and you can find everywhere. The main matter is to decide what to do with this neural network. 

the program show how to classify and image according to CIFAR-10 dataset and use Pytorch. I prefered pytorch because tensor flow has many version and needs to download correct cuDNN and install manually. i did but did not like , so inorder to guarantee program uses GPU and test, i prefered Pytorch. 
program takes long time to run. First it will download CIFAR 10 dataset and uses 10 epoch (you can increase) to train model.
after training finishes, tkinter will open and ask an image to predict. Warning CIFAR-10 does not recognize all image. check CIFAR 10 dataset for details. 
i chose 10 epoch here. but you can try with 30 epoch.

Functions we use in program
Net: Defines the neural network architecture with two convolutional layers and three fully connected layers.
predict_image(img): Given an image, this function processes and feeds it into the trained model, then returns the predicted class label.
get_user_correction(predicted_label): A user interface to correct the prediction made by the model. If the model's prediction is incorrect, users can choose the correct label via a dropdown menu.
open_image(): This function facilitates users to open an image from their local storage. Once an image is uploaded, it displays the prediction on the GUI and allows users to correct it if needed.
so that is . feel free to ask.. 





