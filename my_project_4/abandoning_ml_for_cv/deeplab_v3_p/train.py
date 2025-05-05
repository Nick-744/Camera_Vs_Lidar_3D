# Για να μην έχουμε σφάλμα με το OpenMP!!! Χρειάζεται μόνο στο αρχείο που εκτελείται 1ο!
from os import environ

environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
''' OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program.
That is dangerous, since it can degrade performance or cause incorrect results. The best thing to
do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding
static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented
workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the
program to continue to execute, but that may cause crashes or silently produce incorrect
results. For more information, please see http://www.intel.com/software/products/support/. '''

environ['CUDA_LAUNCH_BLOCKING'] = '1'
# RuntimeError: CUDA error: an illegal memory access was encountered
# CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
# For debugging consider passing CUDA_LAUNCH_BLOCKING=1
# Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

# ----------------------------------------

import torch
import cv2
from os import listdir
from os.path import join, dirname, abspath
import numpy as np
from tqdm import tqdm

from transformer import transform
import my_model

torch.backends.cudnn.enabled = False # Για debugging!

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir, transform = None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.images = sorted(listdir(images_dir))
        self.transform = transform

        return;

    ''' Όταν κάνεις hover πάνω από το Dataset που κληρονομεί η MyDataset,
    βλέπεις ότι πρέπει υποχρεωτικά να υλοποιήσεις (override)
    τις μεθόδους __len__ και __getitem__! '''
    def __len__(self):
        return len(self.images);

    def __getitem__(self, idx):
        img_filename = self.images[idx]
        img_path = join(self.images_dir, img_filename)

        mask_filename = img_filename # Τουλάχιστον, στο dataset μου!
        mask_path = join(self.masks_dir, mask_filename)

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        #print(np.unique(mask))

        ''' Σημαντικό: Αν το batch_size είναι 1 και το spatial μέγεθος της εικόνας
        (μετά από όλα τα strides/pooling του DeepLabV3p) γίνει 1x1, προκαλείται BatchNorm σφάλμα!!!
        Συγκεκριμένα, το σφάλμα είναι: '''
        # File "N:\nick_programs\envs\compgeo2025\Lib\site-packages\torch\nn\functional.py", line 2820, in batch_norm
        #     _verify_batch_size(input.size())
        # File "N:\nick_programs\envs\compgeo2025\Lib\site-packages\torch\nn\functional.py", line 2786, in _verify_batch_size
        #     raise ValueError(
        # ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 256, 1, 1])

        ''' Αυτό μπορεί να συμβεί:
        - είτε αν κάνουμε resize σε πολύ μικρές εικόνες (π.χ. 256x256...)
        - είτε αν το τελευταίο batch έχει μόνο 1 δείγμα!!!
        
        ### Το διόρθωσα βάζοντας drop_last = True στο DataLoader! '''
        # target_height = 256
        # scale = target_height / image.shape[0]
        # Στην αρχή, ήθελα να κρατήσω το aspect ratio... δηλαδή, 1η λύση (μεγαλύτερες εικόνες)!
        new_size = my_model.model_input_size
        image = cv2.resize(image, new_size, interpolation = cv2.INTER_LINEAR)
        mask = cv2.resize(mask, new_size, interpolation = cv2.INTER_NEAREST)

        mapping = {29: 0, 76: 1} # Αλλιώς, πρέπει να έχω 106 κλάσεις..., ενώ θέλω ΜΟΝΟ 3!
        #print(np.unique(mask))  # Για να ξέρω πόσες κλάσεις έχω στο mask!
        mask = np.vectorize(mapping.get)(mask)
        mask = torch.from_numpy(mask).long()

        if self.transform:
            image = self.transform(image)

        return (image, mask);

def train_1_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0

    loop = tqdm(loader, desc = f"Epoch {epoch + 1}", leave = True)
    for (images, masks) in loop:
        (images, masks) = (images.to(device), masks.to(device))
        #print(images.device, masks.device) # Στην μονάδα επεξεργασίας που θα φορτωθεί το
                                            # μοντέλο, πρέπει να φορτωθούν και τα δεδομένα!
        optimizer.zero_grad()
        output = model(images)['out']
        loss = criterion(output, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        loop.set_postfix(loss = loss.item())

    return total_loss / len(loader);

def train_model(epochs, model, loader, optimizer, criterion, device):
    for epoch in range(epochs):
        avg_loss = train_1_epoch(
            model = model,
            loader = loader,
            optimizer = optimizer,
            criterion = criterion,
            device = device,
            epoch = epoch
        )

        print(f"- Epoch {epoch + 1} finished | Avg loss: {avg_loss:.4f}")

    model_name = 'my_road_model.pth'
    torch.save(model.state_dict(), model_name)
    print(f'Το μοντέλο αποθηκεύτηκε με όνομα: {model_name}')

    return;

def main():
    base_path = dirname(abspath(__file__))

    image_path = abspath(join(base_path, 'dataset', 'rgb'))
    mask_path  = abspath(join(base_path, 'dataset', 'masks'))
    dataset = MyDataset(image_path, mask_path, transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size = 16, shuffle = True, drop_last = True)
    # drop_last = True για να μην έχουμε BatchNorm σφάλμα!

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Potato PC ή όχι...;
    model = my_model.get_model(device)
    print(f'Συσκευή εκτέλεσης: {next(model.parameters()).device}\n') # Πρέπει να δώσει cuda:0 για GPU χρήση!

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    train_model(
        epochs = 60,
        model = model,
        loader = loader,
        optimizer = optimizer,
        criterion = criterion,
        device = device
    )

    return;

if __name__ == '__main__':
    main()
