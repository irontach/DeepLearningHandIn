import numpy as np
from scipy.ndimage import rotate

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class RandomRotation:
    def __init__(self, degrees=10, image_size=32, channels=1):
        self.degrees = degrees
        self.image_size = image_size
        self.channels = channels

    def __call__(self, x_batch):

        B = x_batch.shape[0]
        images = x_batch.reshape(B, self.image_size, self.image_size, self.channels)
        

        
        augmented_images = []
        for img in images:
            angle = np.random.uniform(-self.degrees, self.degrees)

            rot_img = rotate(img, angle, reshape=False, order=1, mode='nearest')
            augmented_images.append(rot_img)
            
        augmented_images = np.array(augmented_images)
        
        return augmented_images.reshape(B, -1)
    
class RandomShift:
    
    def __init__(self, shift_range=3, image_size=28, channels=1):
        self.shift_range = shift_range
        self.image_size = image_size
        self.channels = channels

    def __call__(self, x_batch):
        B = x_batch.shape[0]
        images = x_batch.reshape(B, self.image_size, self.image_size)
        
        augmented = np.zeros_like(images)
        
        shifts_y = np.random.randint(-self.shift_range, self.shift_range + 1, size=B)
        shifts_x = np.random.randint(-self.shift_range, self.shift_range + 1, size=B)
        
        for i in range(B):
            dy, dx = shifts_y[i], shifts_x[i]
            img = images[i]
            
            if dy > 0: # Shift down
                src_y_slice = slice(0, self.image_size - dy)
                dst_y_slice = slice(dy, self.image_size)
            elif dy < 0: # Shift up
                src_y_slice = slice(-dy, self.image_size)
                dst_y_slice = slice(0, self.image_size + dy)
            else:
                src_y_slice = slice(0, self.image_size)
                dst_y_slice = slice(0, self.image_size)

            if dx > 0: # Shift right
                src_x_slice = slice(0, self.image_size - dx)
                dst_x_slice = slice(dx, self.image_size)
            elif dx < 0: # Shift left
                src_x_slice = slice(-dx, self.image_size)
                dst_x_slice = slice(0, self.image_size + dx)
            else:
                src_x_slice = slice(0, self.image_size)
                dst_x_slice = slice(0, self.image_size)
                
            augmented[i, dst_y_slice, dst_x_slice] = img[src_y_slice, src_x_slice]

        return augmented.reshape(B, -1)