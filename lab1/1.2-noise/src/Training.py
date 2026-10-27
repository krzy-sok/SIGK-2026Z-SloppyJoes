import gc

from src.DenoiserUModel import DenoiserUmodel
import torch
import torch.nn as nn
import torch.optim as optim


class Trainer:
    def __init__(self, model: DenoiserUmodel , learning_rate, epochs):
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.epochs = epochs
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=7, factor=0.5, threshold=1e-7, threshold_mode='abs')

    def _train_model(self, trainloader):
        print("Training with Adam optimizer...")

        losses = []

        torch.cuda.empty_cache()
        gc.collect()

        for epoch in range(self.epochs):
            running_loss = 0.0
            breakpoint()
            for i, (noisy_images, clean_images) in enumerate(trainloader):
                noisy_images, clean_images = noisy_images.to(self.device), clean_images.to(self.device)

                # outputs = self.model(noisy_images)
                outputs = self.model(noisy_images)
                loss = self.criterion(outputs, clean_images)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                running_loss += loss.item()

                del noisy_images, clean_images, outputs, loss
                torch.cuda.empty_cache()


            epoch_loss = running_loss / len(noisy)
            losses.append(epoch_loss)
            print(f'Epoch {epoch + 1}, Loss: {epoch_loss}, Learning Rate: {self.optimizer.param_groups[0]["lr"]}')

            self.scheduler.step(epoch_loss)
            print(f"Scheduler state: {self.scheduler.state_dict()}")

            torch.cuda.empty_cache()
            gc.collect()
            print(f"Epoch {epoch} finished")

        print('Finished Training')
        return losses