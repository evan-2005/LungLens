import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from azureml.core import Workspace, Datastore, Dataset
from azureml.core.authentication import InteractiveLoginAuthentication



class DatasetGenerator (Dataset):

    def __init__ (self, pathImageDirectory, pathDatasetFile, listImages, labelList, transform, csvFilePath):
    
        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform
        
        
       # for i listImages:
        #    imagesNames.aprint(listImages)
        labelList= self.createLists(listImages)
        print("listImageLabels")
        print(labelList)
        #print(labelList)
        for i,image in enumerate(listImages):
            listImages[i]=pathImageDirectory+"/"+mainImages+"/"+image
            self.listImagePaths.append(listImages[i])
            
            
        for label in labelList :
            self.listImageLabels.append(label)
        
        
    def __getitem__(self, idx):
        import cv2
        import torchvision.transforms as transforms
        
        image_index = self.listImagePaths[idx]
        try:
            img = Image.open(image_index).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224), (0, 0, 0))
            
        labels = self.listImageLabels[idx]
        imageLabel = torch.FloatTensor(labels)
        
        if self.transform is not None:
            imageData = self.transform(img)
        else:
            imageData = transforms.ToTensor()(img)
            
        # Target mask is of shape (14, 224, 224)
        target_mask = np.zeros((14, 224, 224), dtype=np.float32)
        
        try:
            gray = cv2.imread(image_index, cv2.IMREAD_GRAYSCALE)
            gray = cv2.resize(gray, (224, 224))
            _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
            h, w = gray.shape
            lung_mask = np.zeros_like(gray)
            cv2.rectangle(lung_mask, (int(w * 0.15), int(h * 0.15)), (int(w * 0.85), int(h * 0.85)), 255, -1)
            thresh = cv2.bitwise_and(thresh, lung_mask)
            pseudo_mask = (thresh / 255.0).astype(np.float32)
        except Exception:
            pseudo_mask = np.zeros((224, 224), dtype=np.float32)
            
        for c, label in enumerate(labels):
            if label > 0:
                target_mask[c] = pseudo_mask
                
        target_mask_tensor = torch.tensor(target_mask, dtype=torch.float32)
        return imageData, target_mask_tensor, imageLabel
    
    
            
    def __len__(self):
        
        return len(self.listImagePaths)
    
    
            
    def createLists(self,images): #just Images file names not paths
        
        
        #-------------------- SETTINGS: AML WORKSPACE AND DATASTORE
        interactive_auth = InteractiveLoginAuthentication(tenant_id=os.environ['TENANT_ID'])
        ws = Workspace(
                subscription_id=os.environ['SUBSCRIPTION_ID'],
                resource_group=os.environ["RESOURCE_GROUP"],
                workspace_name=os.environ['WORKSPACE_NAME'],
                auth=interactive_auth
            )
        datastore = Datastore.get(ws, datastore_name=os.environ['DATASTORE_NAME'])



        #-------------------- SETTINGS: MOUNTING THE DATASET TO MAKE IT AVAILABLE
        chestist_data = Dataset.get_by_name(ws,os.environ['DATASET_NAME_CSV'])
        mountPoint = chestist_data.mount()
        mountPoint.start()
        mountFolder = mountPoint.mount_point
        #files=os.listdir(mountFolder+"/dataset"+"/images_01") #Need to generalize for the whole dataset
        patientDataFiltered = pd.read_csv(f"{mountFolder}/Data_Entry_2017_v2020 (1).csv", header=0)
        patientDataFiltered = patientDataFiltered.dropna()
        patientDataFiltered=patientDataFiltered[patientDataFiltered.isin(images).iloc[:,0]]
        
        images=patientDataFiltered['Image Index'].tolist()
        patientDataFiltered['Finding Labels']  = patientDataFiltered['Finding Labels'].replace('No Finding', '')
        all_labels = ['Emphysema', 'Hernia', 'Pneumonia', 'Edema', 'Fibrosis', 'Pleural_Thickening', 'Mass', 'Atelectasis', 'Nodule', 'Effusion', 'Infiltration', 'Pneumothorax', 'Consolidation', 'Cardiomegaly']
        print("all_labels",all_labels)
        
        # obtain list of unique diseases
        all_labels = [x for x in all_labels if len(x) > 0]
        #perform one-hot encoding based on diseases extracted
        for c_label in all_labels:
            if len(c_label)> 1: # leave out empty labels
                patientDataFiltered[c_label] = patientDataFiltered['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)
        
        labelPatients=[]
        for index, rows in patientDataFiltered.iterrows():
            labels = []
            for label in all_labels:
                if label in rows:
                    labels.append(int(rows[label]))
                else:
                    labels.append(0)
            labelPatients.append(labels)
        
  
        
        return labelPatients          
        




