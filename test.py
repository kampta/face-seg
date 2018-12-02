from glob import glob
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch
from nets.MobileNetV2_unet import MobileNetV2_unet


# load pre-trained model and weights
def load_model():
    model = MobileNetV2_unet(None).to(args.device)
    state_dict = torch.load(args.pre_trained, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='Semantic Segmentation')

    # Arguments
    parser.add_argument('--data-folder', type=str, default='./data',
                        help='name of the data folder (default: ./data)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='batch size (default: 8)')
    parser.add_argument('--pre-trained', type=str, default=None,
                        help='path of pre-trained weights (default: None)')

    args = parser.parse_args()
    args.device = torch.device("cpu")

    image_files = sorted(glob('{}/*.jp*g'.format(args.data_folder)))
    model = load_model()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    print('Model loaded')
    print(len(image_files), ' files in folder ', args.data_folder)

    fig = plt.figure()
    for i, image_file in enumerate(image_files):
        if i >= args.batch_size:
            break

        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(image)
        torch_img = transform(pil_img)
        torch_img = torch_img.unsqueeze(0)
        torch_img = torch_img.to(args.device)

        # Forward Pass
        logits = model(torch_img)
        mask = np.argmax(logits.data.cpu().numpy(), axis=1)

        # Plot
        ax = plt.subplot(2, args.batch_size, 2 * i + 1)
        ax.axis('off')
        ax.imshow(image.squeeze())

        ax = plt.subplot(2, args.batch_size, 2 * i + 2)
        ax.axis('off')
        ax.imshow(mask.squeeze())

    plt.show()
