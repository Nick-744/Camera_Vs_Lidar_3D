import torch
import cv2
from os import listdir
from os.path import join, dirname, abspath
import numpy as np
from tqdm import tqdm

from my_model import model_input_size

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

        mask_filename = img_filename.replace('_', '_road_', 1)
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
        - είτε αν κάνουμε resize σε πολύ μικρές εικόνες (π.χ. 256x256, όπως εγώ...)
        - είτε αν το τελευταίο batch έχει μόνο 1 δείγμα!!!
        
        ### Το διόρθωσα βάζοντας drop_last = True στο DataLoader! '''
        # target_height = 256
        # scale = target_height / image.shape[0]
        # Στην αρχή, ήθελα να κρατήσω το aspect ratio... δηλαδή, 1η λύση (μεγαλύτερες εικόνες)!
        new_size = model_input_size
        image = cv2.resize(image, new_size, interpolation = cv2.INTER_LINEAR)
        mask = cv2.resize(mask, new_size, interpolation = cv2.INTER_NEAREST)

        mapping = {0: 0, 29: 0, 76: 1, 105: 2} # Αλλιώς, πρέπει να έχω 106 κλάσεις..., ενώ θέλω ΜΟΝΟ 3!
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

from transformer import transform
from my_model import model

def main():
    base_path = dirname(abspath(__file__))

    image_path = abspath(join(base_path, '..', 'KITTI', 'data_road', 'training', 'image_2'))
    mask_path  = abspath(join(base_path, '..', 'KITTI', 'data_road', 'training', 'gt_image_2'))
    dataset = MyDataset(image_path, mask_path, transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size = 4, shuffle = True, drop_last = True)
    # drop_last = True για να μην έχουμε BatchNorm σφάλμα!

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Potato PC ή όχι...;
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    train_model(
        epochs = 20,
        model = model,
        loader = loader,
        optimizer = optimizer,
        criterion = criterion,
        device = device
    )

    return;

if __name__ == '__main__':
    main()
