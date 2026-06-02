import os
import numpy as np
import time
import sys
import re
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tempfile import TemporaryFile
import torch.nn.functional as func
from CNNModel import CNNModel
from DatasetGenerator import DatasetGenerator
from azureml.core import Workspace, Datastore, Dataset
from azureml.core.authentication import InteractiveLoginAuthentication
from sklearn.metrics import roc_auc_score

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1.0):
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()                            
        dice_loss = 1.0 - (2.0 * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)  
        BCE = nn.functional.binary_cross_entropy(inputs, targets, reduction='mean')
        
        return BCE + dice_loss

def dice_coefficient(y_pred, y_true, smooth=1e-6):
    y_pred_bin = (y_pred > 0.5).float()
    intersection = (y_pred_bin * y_true).sum(dim=(2, 3))
    union = y_pred_bin.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()

class TrainerTester ():

    def trainer(pathDirData, pathFileTrain, pathFileVal,pathFileTest, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, transResize, transCrop, launchTimestamp, checkpoint):
    
        #-------------------- SETTINGS: NETWORK ARCHITECTURE
        if nnArchitecture == 'CNNModel': model = CNNModel(nnClassCount, nnIsTrained).cuda()        
        model = torch.nn.DataParallel(model).cuda()
        
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
        files=os.listdir(mountFolder)
        csvFilePath= mountFolder

        #-------------------- SETTINGS: DATA TRANSFORMS (IMAGES SETTINGS)
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.RandomResizedCrop(transCrop))
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)      
        transformSequence=transforms.Compose(transformList)

        #List of images paths for train and validation
        listImagesTrain=[]
        listImagesVal=[]
        listImagesTest=[]
        
        for imagePath in pathFileTrain:
            listImagesTrain.append(os.path.basename(imagePath))
            
        for imagePath in pathFileVal:
            listImagesVal.append(os.path.basename(imagePath))
            
        for imagePath in pathFileTest:
            listImagesTest.append(os.path.basename(imagePath))
        
        labelList=[]
        #-------------------- DATASET BUILDERS
        datasetTrain = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTrain, listImages=listImagesTrain,labelList=labelList, transform=transformSequence,csvFilePath=csvFilePath)
        datasetVal =   DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileVal, listImages=listImagesVal,labelList=labelList, transform=transformSequence,csvFilePath=csvFilePath)
        dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=24, pin_memory=True)
        dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=24, pin_memory=True)
        print("dataset")
        print(list(datasetTrain))
        
        #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam (model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')
                
        #-------------------- SETTINGS: LOSS
        loss = torch.nn.BCELoss(reduction='mean')
        
        #---- Load checkpoint 
        if checkpoint != None:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'],strict=False)
            optimizer.load_state_dict(modelCheckpoint['optimizer'])

        #---- TRAIN THE NETWORK
        lossMIN = 100000
        
        for epochID in range (0, trMaxEpoch):
            
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime
            print("Starting epoch", epochID + 1)
            TrainerTester.epochTrain (model, dataLoaderTrain, optimizer, scheduler, trMaxEpoch, nnClassCount, loss)
            lossVal, losstensor = TrainerTester.epochVal (model, dataLoaderVal, optimizer, scheduler, trMaxEpoch, nnClassCount, loss)
            
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime
            
            scheduler.step(losstensor.item())
            
            if lossVal < lossMIN:
                lossMIN = lossVal    
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, 'm-' + launchTimestamp + '.pth.tar')
                print ('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(lossVal))
            else:
                print ('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(lossVal))
                
    #-------------------------------------------------------------------------------- 
       
    def epochTrain (model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):
        model.train()
        seg_criterion = DiceBCELoss()
      
        print("Training batches...")
        for batchID, (input, target_mask, target) in enumerate (dataLoader):
            target = target.cuda(non_blocking = True)
            target_mask = target_mask.cuda(non_blocking = True)
            input = input.cuda(non_blocking = True)
                
            optimizer.zero_grad()
            class_logits, pred_masks = model(input)
            
            loss_class = loss(class_logits, target)
            loss_seg = seg_criterion(pred_masks, target_mask)
            total_loss = loss_class + 2.0 * loss_seg
                       
            total_loss.backward()
            optimizer.step()
        
    #-------------------------------------------------------------------------------- 
        
    def epochVal (model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):
        with torch.no_grad():
            model.eval()

            lossVal = 0
            lossValNorm = 0
            losstensorMean = 0
            seg_criterion = DiceBCELoss()

            for i, (input, target_mask, target) in enumerate (dataLoader):
                target = target.cuda(non_blocking=True)
                target_mask = target_mask.cuda(non_blocking=True)
                input = input.cuda(non_blocking=True)

                class_logits, pred_masks = model(input)

                loss_class = loss(class_logits, target)
                loss_seg = seg_criterion(pred_masks, target_mask)
                losstensor = loss_class + 2.0 * loss_seg
                
                losstensorMean += losstensor
                lossVal += losstensor.item()
                lossValNorm += 1

            outLoss = lossVal / lossValNorm
            losstensorMean = losstensorMean / lossValNorm

            return outLoss, losstensorMean
        

    
    def computeAUROC (dataGT, dataPRED, classCount):
        
        outAUROC = []
        
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        
        for i in range(classCount):
            try:
                #roc_auc_score(y_true, y_scores)
                outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            except ValueError:
                pass
        print(datanpGT)
        print(datanpPRED)
        
        # save numpy array as csv file
        #from numpy import asarray
        #from numpy import savetxt
        # define data
        #data = asarray([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
        # save to csv file
        #tuple=datanpPRED.shape
        #savetxt('predictions_'+str(tuple[0])+'images'+'.csv', datanpPRED, delimiter=',')
        #ChexnetTrainer.uploadToBlob('predictions_'+str(tuple[0])+'images'+'.csv',"datashowcaseprod")
        
        return outAUROC
    
    #--------------------------------------------------------------------------------  
    

    
    def tester (pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, transResize, transCrop, launchTimeStamp):   
        
        
        #CLASS_NAMES = ['Cardiomegaly', 'Effusion',  'Nodule', 'Pneumonia','Pneumothorax']
        CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia','Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        #CLASS_NAMES = [ 'Effusion', 'Infiltration', 'Mass', 'Nodule','Pneumothorax']
        cudnn.benchmark = True
        
        if nnArchitecture == 'CNNModel': model = CNNModel(nnClassCount, nnIsTrained).cuda()
        
        #import re
        model = torch.nn.DataParallel(model).cuda() 
        print("PWD")
        # !pwd
        checkpoint = torch.load(pathModel)
        state_dict = checkpoint['state_dict']
        remove_data_parallel = False # Change if you don't want to use nn.DataParallel(model)
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(state_dict.keys()):
            match = pattern.match(key)
            new_key = match.group(1) + match.group(2) if match else key
            new_key = new_key[7:] if remove_data_parallel else new_key
            state_dict[new_key] = state_dict[key]
            # Delete old key only if modified.
            if match or remove_data_parallel: 
                del state_dict[key]
        model.load_state_dict(checkpoint['state_dict'], strict=False )
        #optimizer.load_state_dict(checkpoint['optimizer'])
        #modelCheckpoint = torch.load(pathModel)
        #model.load_state_dict(modelCheckpoint['state_dict'])

        #-------------------- SETTINGS: DATA TRANSFORMS, TEN CROPS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        #-------------------- SETTINGS: DATASET BUILDERS
        transformList = []
        transformList.append(transforms.Resize(transResize))
        transformList.append(transforms.TenCrop(transCrop))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        transformSequence=transforms.Compose(transformList)
        
        interactive_auth = InteractiveLoginAuthentication(tenant_id=os.environ['TENANT_ID'])
        ws = Workspace(
                subscription_id=os.environ['SUBSCRIPTION_ID'],
                resource_group=os.environ["RESOURCE_GROUP"],
                workspace_name=os.environ['WORKSPACE_NAME'],
                auth=interactive_auth
            )
        datastore = Datastore.get(ws, datastore_name=os.environ['DATASTORE_NAME'])



        #-------------------- SETTINGS: MOUNTING THE DATASET TO MAKE IT AVAILABLE
        chestist_data = Dataset.get_by_name(ws,os.environ['DATASET_NAME'])
        mountPoint = chestist_data.mount()
        mountPoint.start()
        mountFolder = mountPoint.mount_point
        pathDirData=mountFolder
        #pathDirData=files
        listImagesTest=[]
        for imagePath in pathFileTest:
            listImagesTest.append(os.path.basename(imagePath))
        csvFilePath=""
        labelList=[]
        datasetTest = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTest, listImages=listImagesTest,labelList=labelList, transform=transformSequence,csvFilePath=csvFilePath)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, num_workers=8, shuffle=False, pin_memory=True)
        print("HEY2")
        
        with torch.no_grad():
       
            print(list(datasetTest))
            outGT = torch.FloatTensor().cuda()
            outPRED = torch.FloatTensor().cuda()

            model.eval()

            for i, (input, target_mask, target) in enumerate(dataLoaderTest):

                target = target.cuda()
                outGT = torch.cat((outGT, target), 0)

                bs, n_crops, c, h, w = input.size()

                varInput = input.view(-1, c, h, w).cuda()

                out_class, out_masks = model(varInput)
                outMean = out_class.view(bs, n_crops, -1).mean(1)

                outPRED = torch.cat((outPRED, outMean.data), 0)

            aurocIndividual = TrainerTester.computeAUROC(outGT, outPRED, nnClassCount)
            aurocMean = np.array(aurocIndividual).mean()
            #del dataLoaderTest
            print ('AUROC mean ', aurocMean)

            for i in range (0, len(aurocIndividual)):
                print (CLASS_NAMES[i], ' ', aurocIndividual[i])
        
     
        return

     

