from src.Training import Trainer
from src.ImageDataset import ImageDataset
from torch.utils.data import DataLoader
from src.Unet import Unet

IMG_PATH = "./lab1/1.2-noise/tests/test_images"

def test_epoch_state_changes():
    dataset = ImageDataset(IMG_PATH, 0.03)
    loader = DataLoader(dataset)
    model = Unet()

    trainer = Trainer(model, 0.01, 10)

    old_state = model.state_dict().copy()

    trainer.epoch(loader)
    assert(model.state_dict != old_state)

def test_epoch_los_does_not_grow():
    dataset = ImageDataset(IMG_PATH, 0.03)
    loader = DataLoader(dataset)
    model = Unet()

    trainer = Trainer(model, 0.01, 10)

    loss1 = trainer.epoch(loader)
    loss2 = trainer.epoch(loader)
    assert(loss2 <= loss1)
