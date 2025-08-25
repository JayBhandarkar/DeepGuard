"""
Xception Transforms for Deepfake Detection
Image preprocessing transforms optimized for Xception model input
"""

from torchvision import transforms

# Xception-specific transforms
transform_xception = {
    'train': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'test': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}

# Alternative transforms for different input sizes
transform_xception_224 = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Lightweight transforms for faster inference
transform_xception_fast = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'test': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}

def get_transforms(transform_type='default', input_size=299):
    """
    Get transforms based on type and input size
    
    Args:
        transform_type: Type of transforms ('default', 'fast', 'imagenet')
        input_size: Input size for the model
        
    Returns:
        Dictionary of transforms for train/val/test
    """
    if transform_type == 'fast':
        return transform_xception_fast
    elif transform_type == 'imagenet' or input_size == 224:
        return transform_xception_224
    else:
        return transform_xception

def get_preprocessing_pipeline(transform_type='default', input_size=299):
    """
    Get preprocessing pipeline with additional options
    
    Args:
        transform_type: Type of transforms
        input_size: Input size for the model
        
    Returns:
        Preprocessing pipeline
    """
    base_transforms = get_transforms(transform_type, input_size)
    
    # Add additional preprocessing if needed
    if transform_type == 'enhanced':
        enhanced_transforms = {}
        for split in ['train', 'val', 'test']:
            if split == 'train':
                enhanced_transforms[split] = transforms.Compose([
                    transforms.Resize((input_size, input_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
            else:
                enhanced_transforms[split] = base_transforms[split]
        return enhanced_transforms
    
    return base_transforms
