import torch
from models.basic_cnn import BasicCNN
from models.resnet_model import get_resnet18
from utils.dataset import load_datasets
from utils.train_eval import train_model, evaluate_model, device
from utils.helpers import save_model, load_model
import os

def main():
    train_loader, test_loader, class_names = load_datasets()

    # Create directory to save models
    os.makedirs("saved_models", exist_ok=True)

    # === Train Basic CNN ===
    print("Training Basic CNN...")
    cnn = BasicCNN()
    cnn = train_model(cnn, train_loader)
    save_model(cnn, "saved_models/basic_cnn.pth")
    print("Evaluating Basic CNN:")
    evaluate_model(cnn, test_loader, class_names)

    # === Train ResNet18 ===
    print("\nTraining ResNet18...")
    resnet = get_resnet18()
    resnet = train_model(resnet, train_loader)
    save_model(resnet, "saved_models/resnet18.pth")
    print("Evaluating ResNet18:")
    evaluate_model(resnet, test_loader, class_names)

    # === Load and Evaluate saved model ===
    print("\nLoading saved Basic CNN model...")
    loaded_cnn = load_model(BasicCNN, "saved_models/basic_cnn.pth", device)
    evaluate_model(loaded_cnn, test_loader, class_names)

if __name__ == "__main__":
    main()
