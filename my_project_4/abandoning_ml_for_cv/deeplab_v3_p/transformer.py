from torchvision import transforms

transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Resize((256, 256)),
    # Τελικά μεταφέρθηκε στην μέθοδο __getitem__ της MyDataset,
    # για να γίνει resize και το mask με τον ίδιο τρόπο!
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406], 
        [0.229, 0.224, 0.225]
    )
])
