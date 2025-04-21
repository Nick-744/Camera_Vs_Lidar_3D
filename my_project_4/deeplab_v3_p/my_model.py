from torchvision.models.segmentation import deeplabv3_resnet18
import torch.nn as nn

# Model - DeepLabV3 με MobileNetV3-Large backbone!
weights = None
model = deeplabv3_resnet18(weights = weights)

# - Θέλουμε 2 κλάσεις:
# α) ο δρόμος (και οι 2 λωρίδες)
# β) το υπόλοιπο της εικόνας (εκτός δρόμου)
model.classifier[-1] = nn.Conv2d(256, 2, kernel_size = 1)
# Οπότε, πρέπει να αντικαταστήσουμε το τελευταίο layer {4 ή -1 στο ResNet18/MobileNet} του classifier
# με 1 Conv2d layer που έχει ΜΟΝΟ 2 εξόδους (classes)!
# Επίσης, kernel_size = 1 γιατί θέλουμε να κάνουμε pixel-wise classification!

# Καθορισμός ενιαίου input size για συμβατότητα με το μοντέλο!
model_input_size = (256, 256) # Για να το χρησιμοποιήσουμε στο train.py και στο inference.py!
