# CAIS++_Winter_Project

First name: Zitong
Last name: Huang
email: chuang95@usc.edu

1) Project outline:
   I implemented a multiclass classification project using the American Sign Language (ASL) dataset. The goal was to classify ASL signs into various classes. I used pretrained resnet in this project to make the result more accurate.

2) Dataset:
   I used the ASL dataset for sign language recognition, which is a labelled dataset. Preprocessing involved resizing images, converting into pytorch tensor, and normalizing pixel values to enhance model performance. Resizing is done to ensure uniformity in image sizes, and normalizing process standardizes the pixel values of the images, which helps the model to converge faster. I also defined a custom dataset class that loads images from file paths and applied those transformations.
   I divided the dataset into training dataset, validation dataset, and test dataset. Because there are only few images in the official testing dataset, I randomly selected some labelled image data from the training set, and combine it with the official testing set to increase the number of testing dataset.

3) Model Development and Training:
   I chose the ResNet-50 architecture pretrained on ImageNet to fit this task. Resnet architecture, including resnet50, are known for their depth, which can capture complex hierarchical features. It is pretrained on the large-scale ImageNet dataset. It has learned general features and patterns from diverse images, which can be beneficial for transfer learning to specific tasks like ASL classification.
   In the process of transfer learning, I replaced the final fully connected layer of the ResNet model to match the number of classes in my task, and froze other parameters in the model. This is beacause the lower layers of the original model have learned general features from ImageNet that can be used to our ASL classification task. Freezing these feature extractor layers can prevent from overfitting. However, ASL classification requires to learn some task-specific features, including hand shapes and signs. Therefore, we need to Fine-tune the fully connected layer.
   After that, I identified and grouped the parameters that require gradient updates and used them to define the optimizer. This ensures that only the parameters in the final fully connected layer are updated during training. I chose Adam optimizer and Crossentropyloss function. The number of epoch, batch size, as well as the learning rates are hyperparameters that I can controll.

4) Model Evaluation/Results:
   I evaluated my result by giving all the images into the trained model. I used accuracy, Micro F1 Score, and Macro F1 Score to evaluate the performance of my model.
   Finally, the Average Accuracy: 92.61511961584317, Average micro f1 score: 0.9286369193154034, Average macro f1 score: 0.8910498969748554.

5) Discussion:
a) How well does your dataset, model architecture, training procedures, and chosen metrics fit the task at hand?
  I think my pretrained model architecture fit my dataset well, since Resnet50 is a deep CNN architecture, which can understand the intricate patterns present in images. It has trained in large scale image data before, which can be beneficial when we do transfer learning in limited ASL dataset.
  Also, I froze some parameters, but change and learn a new fully connected layer. I also change some hyperparameters, but since the limited time and calculating capacity, I do not have time to adjust those hhyperparameters.
  For chosen metrics, I chose accuracy, micro F1 score, and macro F1 score. These metrics are suitable for classification tasks, providing a comprehensive understanding of the model's performance.

b) Can your efforts be extended to wider implications, or contribute to social good? Are there any limitations in your methods that should be considered before doing so?
  Yes. I think it can be integrated into applications or devices to assist individuals with hearing impairments. This can enhance accessibility and inclusivity by providing a means for communication through ASL. The application should have an interface that allows users to input gestures and convert it into text or speech. 
  However, the accuracy of this model still needs to be increased. We need to consider more on variations in hand shapes, positions, and lighting conditions. 


Reference: https://www.kaggle.com/code/julichitai/asl-alphabet-classification-pytorch

c) If you were to continue this project, what would be your next steps?
  I would like to increase the performance of this model first. I will use data augmentation to increase the diversity of the dataset, and try different hyperparameters. 
