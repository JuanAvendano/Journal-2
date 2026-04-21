from src.data.dataloader import get_dataloaders
from src.utils.io_utils import load_config
import matplotlib.pyplot as plt

root= r"C:/Users/jcac/OneDrive - KTH/Python/CNN/Journal-2/"
config = load_config(root+"configs/train_config.yaml")
loaders = get_dataloaders(config)

images, labels, paths = next(iter(loaders["train"]))

fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i, ax in enumerate(axes.flat):
    img = images[i].permute(1, 2, 0).numpy()
    # Undo normalisation so the image looks natural
    img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    img = img.clip(0, 1)
    ax.imshow(img)
    ax.axis("off")
plt.tight_layout()
plt.show()