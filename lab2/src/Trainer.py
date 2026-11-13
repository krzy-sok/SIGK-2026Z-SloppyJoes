import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.functional.image import structural_similarity_index_measure

class Trainer:
    def __init__(self, model: nn.Module, learning_rate, epochs, force_gpu=False):
        self.model = model
        if force_gpu and not torch.cuda.is_available():
            raise RuntimeError("Wymuszono użycie GPU (--gpu), ale CUDA nie jest dostępna!")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Device used: {self.device}")
        self.model.to(self.device)
        self.epochs = epochs
        self.citerion = nn.L1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-5)

    def train_model(self, trainloader):
        print("Training with Adam optimizer...")

        losses = []

        torch.cuda.empty_cache()
        gc.collect()

        for epoch in range(self.epochs):
            epoch_loss = self.epoch(trainloader, epoch)
            losses.append(epoch_loss)
            print(f"Epoch {epoch+1} finished")
            print(f'Epoch {epoch + 1}, Loss: {epoch_loss}, Learning Rate: {self.optimizer.param_groups[0]["lr"]}')

        print('Finished Training')
        return losses

    def epoch(self, trainloader, epoch_num):
        running_loss = 0.0
        for i, (clean_image, noisy_image) in enumerate(trainloader):
            noisy_image, clean_image = noisy_image.to(self.device), clean_image.to(self.device)
            
            self.optimizer.zero_grad()

            pred = self.model(noisy_image)
            # breakpoint()
            outputs = torch.clamp(noisy_image + pred, 0.0, 1.0)
            # outputs = pred

            l1_loss = self.citerion(outputs, clean_image)
            ssim_loss = 1.0 - structural_similarity_index_measure(outputs, clean_image, data_range=1.0)
            loss = 0.8 * l1_loss + 0.2 * ssim_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )
            self.optimizer.step()

            running_loss += loss.item()

            # del noisy_image, clean_image, outputs, loss
        # breakpoint()
        print(
            f"Prediction min: {outputs.min().item():.4f}, "
            f"Prediction max: {outputs.max().item():.4f}"
        )
        epoch_loss = running_loss / len(trainloader)
        self.scheduler.step()
        # torch.cuda.empty_cache()
        # gc.collect()
        return epoch_loss