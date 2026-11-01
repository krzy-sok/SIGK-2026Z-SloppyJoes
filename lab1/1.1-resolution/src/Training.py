import gc
import torch
import torch.nn as nn
import torch.optim as optim
from src.SuperResolutionModel import SuperResolutionUModel
from . import Losses

class Trainer:
    def __init__(self, model: SuperResolutionUModel, learning_rate: float, epochs: int):
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.epochs = epochs
        self.criterion = nn.MSELoss()
        self.charbonnier = Losses.CharbonnierLoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            "min",
            patience=7,
            factor=0.5,
            threshold=1e-7,
            threshold_mode="abs",
        )

    def _train_model(self, trainloader, save_path: str | None = None) -> list[float]:
        device_info = (
            f"{self.device} ({torch.cuda.get_device_name(0)})"
            if self.device.type == "cuda"
            else str(self.device)
        )
        print(f"Training with Adam optimizer on device: {device_info}...")

        losses: list[float] = []
        self.model.train()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        for epoch in range(self.epochs):
            running_loss = 0.0

            for low_res_image, clean_image in trainloader:
                low_res_image = low_res_image.to(self.device, non_blocking=True)
                clean_image = clean_image.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                outputs = self.model(low_res_image)

                mse_loss = self.criterion(outputs, clean_image)
                charb_loss = self.charbonnier(outputs, clean_image)
                loss = mse_loss + charb_loss

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(trainloader)
            losses.append(epoch_loss)
            print(
                f"Epoch {epoch + 1}/{self.epochs}, "
                f"Loss: {epoch_loss:.6f}, "
                f"Learning Rate: {self.optimizer.param_groups[0]['lr']}"
            )

            self.scheduler.step(epoch_loss)

            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        if save_path is not None:
            self.save_model(save_path)

        print("Finished Training")
        return losses

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        print(f"Model loaded from {path}")
