import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image

def generate_annotations_file(img_dir='datasets', annots_save_path='./annotation_file.csv'):
    '''
        Создаёт файл с аннотациями и файл с порядком названиями классов
        
        Arguments
            :img_dir: - каталог с папками с картинками
            :annots_save_path: - название файла с аннотациями
    '''
    
    folder_list = os.listdir(img_dir)
    # избавляемся от .zip файлов в списке
    [folder_list.remove(folder) for folder in folder_list if folder.endswith('.zip')]
    
    # сохраняем имена классов
    with open('names.txt', 'w') as names:
        for name in folder_list:
            names.write(str(name)+'\n')
    names.close()

    # готовим файл с аннотациями
    annotations = []
    for class_idx, folder in enumerate(folder_list):
        pwd = os.path.join(img_dir, folder)
        img_list = os.listdir(pwd)
        for img_name in img_list:
            img_path = os.path.join(folder, img_name)
            annotations.append([img_path, class_idx])
    labels = pd.DataFrame(annotations, columns=['img_path', 'label'])
    labels.to_csv(annots_save_path, index=False)


class FlowersDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return image, label