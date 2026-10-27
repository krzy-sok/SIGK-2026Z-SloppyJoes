from src.ImageDataset import ImageDataset
from src.DenoiserUModel import DenoiserUmodel
from torch.utils.data import DataLoader
from src.Training import Trainer
import matplotlib.pyplot as plt

def main():
    model = DenoiserUmodel()
    dataset =  ImageDataset("./images/DIV2K_train_LR_bicubic_X2", 0.03)
    trainer = Trainer(model, 0.01, 50)
    dataLoader = DataLoader(dataset, batch_size=32, shuffle=True)
    breakpoint()
    losses = trainer._train_model(dataLoader)
    breakpoint()
    plt.plot([x for x in range(len(losses))], losses)
    plt.savefig("./loss.png")
    plt.show()


if __name__ == "__main__":
    main()
