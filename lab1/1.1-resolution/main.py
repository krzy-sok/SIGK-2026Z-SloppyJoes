import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.Evaluation import Evaluator
from src.ImageDataset import ImageDataset
from src.SuperResolutionModel import SuperResolutionUModel
from src.Training import Trainer

TRAIN_DIR = "../../images/train"
VALIDATE_DIR = "../../images/validate"
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001

def train_or_load_model(low_res_size: int, force_retrain: bool = False) -> SuperResolutionUModel:
    model = SuperResolutionUModel(output_size=256, use_residual=True)
    trainer = Trainer(model, learning_rate=LEARNING_RATE, epochs=EPOCHS)
    weights_path = f"./model_{low_res_size}x{low_res_size}.pth"

    if os.path.exists(weights_path) and not force_retrain:
        print(f"\n[Info] Found existing weights at {weights_path} – training skipped.")
        trainer.load_model(weights_path)
        return model

    print(f"\n=== Model training for input resolution {low_res_size}x{low_res_size} -> 256x256 ===")
    dataset = ImageDataset(TRAIN_DIR, low_res_size=low_res_size, max_images=200)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    losses = trainer._train_model(dataloader, save_path=weights_path)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker="o", markersize=3)
    plt.title(f"Loss function (Super-Resolution {low_res_size}x{low_res_size} -> 256x256)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE + Charbonnier)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_plot_path = f"./loss_{low_res_size}x{low_res_size}.png"
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()
    print(f"Loss function graph saved to {loss_plot_path}")

    return model

def main():
    # model training for 32x32 and 64x64
    model_32 = train_or_load_model(low_res_size=32)
    model_64 = train_or_load_model(low_res_size=64)
    
    print("\n=== Evaluation DIV2K ===")
    evaluator = Evaluator()

    results_32 = evaluator.evaluate_resolution(VALIDATE_DIR, model_32, low_res_size=32)
    results_64 = evaluator.evaluate_resolution(VALIDATE_DIR, model_64, low_res_size=64)

    all_results = {**results_32, **results_64}
    table_str = evaluator.format_results_table(all_results)

    print("\n" + table_str + "\n")

    with open("./results_table.txt", "w", encoding="utf-8") as f:
        f.write(table_str + "\n")
    print("Table of results saved to ./results_table.txt")

    # save visual comparison
    evaluator.save_visual_comparison(
        VALIDATE_DIR,
        model_32=model_32,
        model_64=model_64,
        output_path="./comparison_results.png",
        image_idx=0,
    )

if __name__ == "__main__":
    main()
