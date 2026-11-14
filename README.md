\# Robust Multi-Class Brain Tumor Classification Using Deep Learning and MRI Images



Course project for \[add course name]. By \[your name].



This project classifies brain MRI images into four classes (\*\*Glioma, Meningioma, Pituitary tumor, No tumor\*\*) using a fine-tuned EfficientNet-B0. It then tests how the model behaves when the images are noisy or blurred.



> \*\*Disclaimer:\*\* this is a school project. It is not a medical tool and must not be used for real diagnosis.



\## Dataset



\- Source: \[add the dataset name and link here]

\- 7,200 images: 5,600 in the original training part and 1,600 in the test part (400 per class)

\- My split: the training part was divided 80/20 (stratified, random\_state=42) into 4,480 training and 1,120 validation images. The 1,600 test images were not used for any decision.

\- The images are \*\*not\*\* included in this repository. Download them from the link above and change the paths in the notebook.



\## What I did



1\. \*\*Data checks:\*\* class counts, image sizes, file integrity and a stratified split.

2\. \*\*Shortcut check:\*\* a classifier that only sees the image width and height (no pixels) reached 45.09% (logistic regression) and 50.62% (decision tree), compared with 25% for random guessing. So image shape carries class information, and black padding bars could act as a shortcut.

3\. \*\*Model comparison:\*\* five pretrained models (MobileNetV3-Small, EfficientNet-B0, ResNet-18, ResNet-34, ResNet-50) were compared with three resize options (black padding, direct resize, center crop). I used frozen features with logistic regression and looked at accuracy, macro precision, macro recall and macro F1, not only accuracy. EfficientNet-B0 ranked first. I chose \*\*center crop\*\*, because it performed as well as padding (McNemar p = 0.7539) and it removes the black bars.

4\. \*\*Normalization:\*\* ImageNet normalization (96.07% validation accuracy) was better than dataset-specific normalization (94.20%, McNemar p = 0.0139), so ImageNet values were kept.

5\. \*\*Training:\*\* two stages with early stopping on validation loss (patience 5, minimum improvement 0.001).

&#x20;  - Stage 1: only the new classification layer (best epoch 12, validation accuracy 93.75%)

&#x20;  - Stage 2: the last three EfficientNet-B0 blocks fine-tuned (best epoch 9, validation accuracy 97.59%, macro F1 0.9759)

6\. \*\*Clean test evaluation\*\* and \*\*robustness tests\*\* with Gaussian noise and Gaussian blur.



\## Results



\### Clean test set (1,600 images)



| Metric | Value |

|---|---|

| Overall accuracy | 94.38% (95% CI 93.14 to 95.40) |

| Macro F1 | 0.9427 |

| Missed tumors (tumor classified as No tumor) | 39 of 1,200 (3.25%) |

| False alarms (No tumor classified as a tumor) | 0 of 400 |



| Class | Recall (400 images each) |

|---|---|

| Glioma | 82.25% |

| Meningioma | 96.25% |

| Pituitary | 99.00% |

| No tumor | 100% |



Glioma is the weakest class: 71 of the 90 mistakes were Glioma images (36 labeled Meningioma, 34 labeled No tumor, 1 labeled Pituitary).



\### Robustness (same 1,600 test images with noise or blur added)



| Condition | Accuracy |

|---|---|

| Clean | 94.38% |

| Noise 0.05 | 85.81% |

| Noise 0.10 | 75.06% |

| Noise 0.20 | 33.25% |

| Noise 0.30 | 26.75% |

| Blur 1 | 89.69% |

| Blur 2 | 44.94% |

| Blur 3 | 35.88% |

| Blur 5 | 28.50% |



Every drop is statistically significant compared with the clean images (McNemar p < 0.0001). With mild distortion the model mixes up the tumor types, mainly Glioma. With strong distortion it labels almost every image as No tumor, which means tumors are missed.



\## Limitations



\- The test accuracy (94.38%) is lower than the validation accuracy (97.59%).

\- Glioma recall is low (82.25%).

\- Center crop removes more area from No tumor images than from tumor images, and cropping does not fully remove aspect-ratio information, so a weak shape shortcut may remain.

\- The model comparison used frozen features with logistic regression, so the ranking could change after fine-tuning.

\- Noise and blur were simple Gaussian distortions, with one noise sample per level and a fixed seed (42). They are not the same as real scanner noise.

\- The model was not trained with noisy or blurred images, and I did not test whether that would help.

\- One dataset only, with no external validation.



\## Repository structure



```

brain-tumor-classification/

├── brain\_tumor\_classification.ipynb   # all the code, results and explanations

├── models/

│   ├── stage1\_best.pt                 # after stage 1 (classification layer only)

│   └── stage2\_best.pt                 # FINAL model (fine-tuned EfficientNet-B0)

├── results/                           # saved training history and predictions (json)

├── requirements.txt

└── README.md

```



\## How to run



1\. Install the libraries: `pip install -r requirements.txt`

2\. Download the dataset and set the paths in the notebook.

3\. In the "Saving and Reloading Results" cell, change `saved\_dir` to the folder where you keep `models/` and `results/`.

4\. Run the notebook from the top. Training and evaluation cells check the saved files first and load them, so nothing is trained again unless you delete the saved files. On my CPU the training took more than an hour.



\## Use the trained model on one image



```python

import torch

import torch.nn as nn

from torchvision import models, transforms

from PIL import Image



classes = \["Glioma", "Meningioma", "Pituitary", "Notumor"]



\# rebuild the model and load the final weights

model = models.efficientnet\_b0(weights=None)

model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(1280, 4))

model.load\_state\_dict(torch.load("models/stage2\_best.pt", map\_location="cpu"))

model.eval()



\# same preprocessing as the test images: center crop, resize, ImageNet normalization

def center\_crop\_square(image):

&#x20;   w, h = image.size

&#x20;   side = min(w, h)

&#x20;   left = (w - side) // 2

&#x20;   top = (h - side) // 2

&#x20;   return image.crop((left, top, left + side, top + side)).resize((224, 224))



transform = transforms.Compose(\[

&#x20;   transforms.Lambda(center\_crop\_square),

&#x20;   transforms.ToTensor(),

&#x20;   transforms.Normalize(\[0.485, 0.456, 0.406], \[0.229, 0.224, 0.225])

])



image = Image.open("path/to/image.jpg").convert("RGB")

tensor = transform(image).unsqueeze(0)



with torch.no\_grad():

&#x20;   probabilities = torch.softmax(model(tensor), dim=1)\[0]



print(classes\[int(probabilities.argmax())], probabilities.tolist())

```



The model only knows these four classes, so it always returns one of them, even for an image that is not a brain MRI.



\## Author



Abdulrazaq, Georgia Southern University, 2026

