from scipy.ndimage import binary_erosion
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import binary_opening
from skimage.measure import label, regionprops

def pprocess(seg, mask, skin_gap, area_th, pixel_size):
    
    # Post process density segmentation
    # Convert parameters to pixels
    
    skin_gap = round(2*skin_gap/pixel_size)
    area_threshold = round(area_th/(pixel_size**2))
    
    # Erode breast mask
    
    maske = binary_erosion(mask, generate_binary_structure(2, 2*skin_gap))
    
    # Apply area filter
    
    # Realizar la operación lógica AND entre seg y maske
    seg_and_maske = seg & maske

    # Crear un elemento estructurante para la operación de apertura
    structure = generate_binary_structure(2, 1)

    # Aplicar la operación de apertura a la imagen binaria seg_and_maske
    opened = binary_opening(seg_and_maske, structure)

    # Eliminar áreas pequeñas de la imagen binaria utilizando el umbral de área
    labeled = label(opened)
    for region in regionprops(labeled):
        
        if region.area < area_threshold:
            
            opened[labeled == region.label] = 0
        
    # Asignar la imagen resultante a la variable seg
    seg = opened
    
    return seg
