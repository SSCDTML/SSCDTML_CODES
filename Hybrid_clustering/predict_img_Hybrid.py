import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from utils.data_loading import BreastSegDataset
from unet import UNet

def predict_img_from_matlab(img_path:str, cnn_checkpoint:str ='data/pytorch_checkpoints/checkpoint_epoch21.pth'):

    net = UNet(n_channels=1, n_classes=2, bilinear=False) #crea una instancia de la red
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net.to(device=device) #envía la red al dispositivo especificado anteriormente
    net.load_state_dict(torch.load(cnn_checkpoint, map_location=device)) #carga el archivo de control previamente entrenado
    full_img = Image.open(img_path) # carga la imagen  en formato .png y la convierte en un objeto PIL
    
    out_threshold=0.5 #se establece un umbral para la salida binaria de la red
    net.eval() #establece la red en modo evaluación, es decir, en modo inferencia
    img = np.asarray(full_img) # se convierte la imagen en un arreglo numpy
    img = BreastSegDataset.preprocess(img, is_mask=False)
    img = torch.from_numpy(img)
    img = img.unsqueeze(0) #agrega una dimensión adicional al tensor
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad(): #establece que no se calcularán los gradientes
        output = net(img) #se aplica la imagen a la red y se obtiene la salida

        if net.n_classes > 1: # verifica que la red esté diseñada para clasificar más de una clase
            probs = F.softmax(output, dim=1) # # si es así, se utiliza softmax
            probs = probs[0] #se selecciona la primera dimensión, ya que se trabaja con una sola imagen

        else: #si tiene una sola clase...
            probs = torch.sigmoid(output)[0]

         #se definen transformaciones que se aplicarán a la máscara de salida
        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze() #máscara 2D con valores entre 0 y 1

    if net.n_classes == 1:
        return mask_to_matrix((full_mask > out_threshold).numpy())
    
    else:
        full_mask = F.one_hot(full_mask.argmax(dim=0), net.n_classes)
        full_mask = full_mask.permute(2, 0, 1).numpy()

        return mask_to_matrix(full_mask[0])

def mask_to_matrix(mask: np.ndarray):
    if mask.ndim == 2:
        return (mask * 255).astype(np.uint8)
    elif mask.ndim == 3:
        return (np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8)