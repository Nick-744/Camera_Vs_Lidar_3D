from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
import torch.nn as nn

# Model - DeepLabV3 με MobileNetV3-Large backbone!
weights = None
model = deeplabv3_mobilenet_v3_large(weights = weights)

# - Θέλουμε 3 κλάσεις:
# α) ο δρόμος (και οι 2 λωρίδες, αν είναι "ενωμένες")
# β) η αντίθετη λωρίδα, αν είναι "διαχωρισμένες"
# γ) το υπόλοιπο της εικόνας (εκτός δρόμου)
model.classifier[4] = nn.Conv2d(256, 3, kernel_size = 1)
# Οπότε, πρέπει να αντικαταστήσουμε το τελευταίο layer του classifier
# με 1 Conv2d layer που έχει ΜΟΝΟ 3 εξόδους (classes)!
# Επίσης, kernel_size = 1 γιατί θέλουμε να κάνουμε pixel-wise classification!

# Καθορισμός ενιαίου input size για συμβατότητα με το μοντέλο!
model_input_size = (256, 256) # Για να το χρησιμοποιήσουμε στο train.py και στο inference.py!
