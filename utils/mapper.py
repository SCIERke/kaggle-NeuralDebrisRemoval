from typing_extensions import dataclass_transform

def calculate_pixel_per_stride(layer ,img) -> float:
    _b, _c, h, w = layer.shape
    
    img_h , img_w, _= img.shape
    w_n_strides = img_w / float(w)
    h_n_strides = img_h / float(h)
    
    return (w_n_strides, h_n_strides)
    

def annotation_to_layer_corr(annotation ,layer ,img):
    x, y, w, h = annotation[0]['bbox']
    
    w_n_strides, h_n_strides = calculate_pixel_per_stride(layer ,img)
    
    return (x/w_n_strides, y/h_n_strides, w/w_n_strides, h/h_n_strides)
