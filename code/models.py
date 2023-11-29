import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, vit_b_16, convnext_tiny, densenet121, efficientnet_v2_s,maxvit_t, swin_t
import torchvision.models as models
from torchvision.models import VisionTransformer
# from  torchvision.models import ConvNeXt
# from torchvision.models.detection import ConvNeXt
# torchvision.models.vit_b_16(*, weights: Optional[ViT_B_16_Weights] = None, progress: bool = True, **kwargs: Any) 
from paths import OUTPUT_DIR


def boolCheck(input):
    if input == "True":
        return True 
    elif input == "False":
        return None

#swin_t
def get_swin_t_model(num_classes, device_index, to_use_pre_trained_weights) :
    device = torch.device(device_index) if torch.cuda.is_available() else torch.device('cpu')
    np.random.seed(31101995)
    torch.manual_seed(31101995)
    to_use_weight = boolCheck(to_use_pre_trained_weights)
    model = swin_t( num_classes=num_classes)
    model.fc = nn.Linear(1024,num_classes)
    # print(model.summary)
    print(model)
    return model

def get_vit_b_16_model(num_classes, device_index, to_use_pre_trained_weights) :
    device = torch.device(device_index) if torch.cuda.is_available() else torch.device('cpu')
    np.random.seed(31101995)
    torch.manual_seed(31101995)
    to_use_weight = boolCheck(to_use_pre_trained_weights)
    model = vit_b_16( num_classes=num_classes)
    model.fc = nn.Linear(1024,num_classes)
    # print(model.summary)
    print(model)
    return model

def get_maxvit_t_model(num_classes, device_index, to_use_pre_trained_weights) :
    device = torch.device(device_index) if torch.cuda.is_available() else torch.device('cpu')
    np.random.seed(31101995)
    torch.manual_seed(31101995)
    to_use_weight = boolCheck(to_use_pre_trained_weights)
    model = maxvit_t(pretrained=True)
    model.fc = nn.Linear(1024,num_classes)
    return model

def get_densenet121_model(num_classes, device_index, to_use_pre_trained_weights) :
    device = torch.device(device_index) if torch.cuda.is_available() else torch.device('cpu')
    np.random.seed(31101995)
    torch.manual_seed(31101995)
    to_use_weight = boolCheck(to_use_pre_trained_weights)
    model =  densenet121(pretrained=True)
    # model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    n_inputs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(n_inputs, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(64, num_classes),
    )
    return model

def get_convnext_tiny_model(num_classes, device_index, to_use_pre_trained_weights) :
    device = torch.device(device_index) if torch.cuda.is_available() else torch.device('cpu')
    np.random.seed(31101995)
    torch.manual_seed(31101995)
    to_use_weight = boolCheck(to_use_pre_trained_weights)
    model =  convnext_tiny(pretrained=True)
    model.fc = nn.Linear(1024,num_classes)
    # model = convnext_tiny(weights=True, num_classes=num_classes)
    # model = ConvNeXt(weights=True, num_classes=num_classes)
    # model = CustomMaxvit_t(num_classes=num_classes)
    return model


def get_efficientnet_v2_s_model(num_classes, device_index, to_use_pre_trained_weights):
    device = torch.device(device_index) if torch.cuda.is_available() else torch.device('cpu')    
    np.random.seed(31101995)
    torch.manual_seed(31101995)
    to_use_weight = boolCheck(to_use_pre_trained_weights)
    model =  efficientnet_v2_s(pretrained=True)
    
    # Change the final classification head.
    # model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    # model._fc= torch.nn.Linear(in_features=model._fc.in_features, out_features=num_classes, bias=True)

    model.classifier[-1]= torch.nn.Linear(in_features=model.classifier[-1].in_features, out_features=num_classes, bias=True)
    
    
    # model.fc = nn.Linear(1024,num_classes)
    # print(model)
    return model


def get_mobilenet_v2_model(num_classes, device_index, to_use_pre_trained_weights):
    device = torch.device(device_index) if torch.cuda.is_available() else torch.device('cpu')    
    np.random.seed(31101995)
    torch.manual_seed(31101995)
    to_use_weight = boolCheck(to_use_pre_trained_weights)
    #mobilenet_v2
    mobilenet = models.mobilenet_v2(pretrained=True)
    mobilenet.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=False),
                    nn.Linear(in_features=mobilenet.classifier[1].in_features, out_features=num_classes, bias=True))
    return mobilenet


def get_inception_v3_model(num_classes, device_index, to_use_pre_trained_weights):
    device = torch.device(device_index) if torch.cuda.is_available() else torch.device('cpu')    
    np.random.seed(31101995)
    torch.manual_seed(31101995)
    to_use_weight = boolCheck(to_use_pre_trained_weights)
    #inception_v3
    inception = models.inception_v3(pretrained=True)
    inception.fc =  nn.Linear(in_features=inception.fc.in_features, out_features=num_classes, bias=True)
    return inception


def get_resnet18_model(num_classes, device_index, to_use_pre_trained_weights) :
    device = torch.device(device_index) if torch.cuda.is_available() else torch.device('cpu')
    np.random.seed(31101995)
    torch.manual_seed(31101995)
    to_use_weight = boolCheck(to_use_pre_trained_weights)
    model =  resnet18(weights=to_use_weight)
    # model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    n_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(n_inputs, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.6),
        nn.Linear(64, num_classes),
    )
    return model

def get_resnet50_model(num_classes, device_index, to_use_pre_trained_weights):
    device = torch.device(device_index) if torch.cuda.is_available() else torch.device('cpu')
    np.random.seed(31101995)
    torch.manual_seed(31101995)
    to_use_weight = boolCheck(to_use_pre_trained_weights)
    model = resnet50(weights=to_use_weight)
    # model = resnet50(weights=to_use_pre_trained_weights)
    n_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(n_inputs, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(64, num_classes),
    )
    return model

def get_teacher_student_model(num_classes, device_index, mod, to_use_pre_trained_weights, teacher_name, student_name, fold):
    device = torch.device(device_index) if torch.cuda.is_available() else torch.device('cpu')
    #get pre-trained teacher model
    #initialize teacher model 
    # teacher_model_path = OUTPUT_DIR / mod / 'baseline/resnet50/resnet50_base.pt'
    teacher = get_model_by_name(teacher_name,num_classes,  mod,device_index, to_use_pre_trained_weights )
    teacher_model_path = OUTPUT_DIR / mod / 'baseline' / teacher_name  /f'fold{fold}' / f'{teacher_name}_base_fold_{fold}.pt' 
    teacher.load_state_dict(torch.load(teacher_model_path))
    #initialize student model
    student =  get_model_by_name(student_name,num_classes,  mod,device_index, to_use_pre_trained_weights )
    return  teacher, student



# def get_teacher_student_model(num_classes, device_index, mod, to_use_pre_trained_weights):
#     device = torch.device(device_index) if torch.cuda.is_available() else torch.device('cpu')
#     #get pre-trained teacher model
#     #initialize teacher model 
#     # teacher = get_resnet50_model(num_classes, device_index, to_use_pre_trained_weights)
#     # teacher_model_path = OUTPUT_DIR / mod / 'baseline/resnet50_base_no_weights/resnet50_base.pt'
#     # teacher_model_path = OUTPUT_DIR / mod / 'baseline/resnet50/resnet50_base.pt'

#     teacher = get_efficientnet_v2_s_model(num_classes, device_index, to_use_pre_trained_weights)
#     teacher_model_path = OUTPUT_DIR / mod / 'baseline/efficientnet_v2_s/efficientnet_v2_s_base.pt' 
#     teacher.load_state_dict(torch.load(teacher_model_path))

#     #initialize student model
#     # student = get_resnet18_model(num_classes, device_index, to_use_pre_trained_weights)
#     student = get_densenet121_model(num_classes, device_index, to_use_pre_trained_weights)
#     return  teacher, student

def get_student_teacher_model(num_classes, device_index, mod, to_use_pre_trained_weights):
    device = torch.device(device_index) if torch.cuda.is_available() else torch.device('cpu')
    #get pre-trained student  model
    teacher = get_resnet18_model(num_classes, device_index, to_use_pre_trained_weights)
    teacher_model_path = OUTPUT_DIR / mod / 'baseline/resnet18/resnet18_base.pt'
    teacher.load_state_dict(torch.load(teacher_model_path))
    #Initialize teacher(ResNet50 models)
    student = get_resnet50_model(num_classes, device_index, to_use_pre_trained_weights)
    return  teacher, student

def get_teacher_teacher_model(num_classes, device_index, mod,  to_use_pre_trained_weights):
    device = torch.device(device_index) if torch.cuda.is_available() else torch.device('cpu')
    #get pre-trained teacher model
    #initialize teacher model 
    pre_trained_teacher = get_resnet50_model(num_classes, device_index, to_use_pre_trained_weights)
    # teacher_model_path = OUTPUT_DIR / mod / 'baseline/resnet50_base_no_weights/resnet50_base.pt'
    pre_trained_teacher_model_path = OUTPUT_DIR / mod / 'baseline/resnet50/resnet50_base.pt'
    pre_trained_teacher.load_state_dict(torch.load(pre_trained_teacher_model_path))
    #Initialize teacher(ResNet50 model)
    teacher = get_resnet50_model(num_classes, device_index, to_use_pre_trained_weights)
    return pre_trained_teacher, teacher

def get_student_student_model(num_classes, device_index, mod,  to_use_pre_trained_weights):
    device = torch.device(device_index) if torch.cuda.is_available() else torch.device('cpu')
    #get pre-trained student  model
    pre_trained_student = get_resnet18_model(num_classes, device_index, to_use_pre_trained_weights)
    pre_trained_student_model_path = OUTPUT_DIR / mod / 'baseline/resnet18/resnet18_base.pt'
    pre_trained_student.load_state_dict(torch.load(pre_trained_student_model_path))
    #initialize student model
    student = get_resnet18_model(num_classes, device_index, to_use_pre_trained_weights)
    return pre_trained_student, student

def get_bidirectional_teacher_student_model(num_classes, device_index, mod,  to_use_pre_trained_weights):



    
    return 

def get_model_by_name(model_name,num_classes,  mod,device_index, to_use_pre_trained_weights ):
    if model_name is not None:
        if model_name == 'resnet18':
            model = get_resnet18_model(num_classes, device_index, to_use_pre_trained_weights)
        elif model_name == 'resnet50':
            model = get_resnet50_model(num_classes, device_index, to_use_pre_trained_weights) 
        elif model_name == 'convnext_tiny':
            model = get_convnext_tiny_model(num_classes, device_index, to_use_pre_trained_weights)  
        elif model_name == 'efficientnet_v2_s':
            model = get_efficientnet_v2_s_model(num_classes, device_index, to_use_pre_trained_weights)
        elif model_name =='maxvit_t':
            model = get_maxvit_t_model(num_classes, device_index, to_use_pre_trained_weights)
        elif model_name=='swin_t':
            model= get_swin_t_model(num_classes, device_index, to_use_pre_trained_weights)
        elif model_name=='vit_b_16':
            model= get_vit_b_16_model(num_classes, device_index, to_use_pre_trained_weights)
        elif model_name == 'densenet121':
            model = get_densenet121_model(num_classes, device_index, to_use_pre_trained_weights) 
        elif model_name == 'mobilenet_v2':
            model = get_mobilenet_v2_model(num_classes, device_index, to_use_pre_trained_weights)
        elif model_name == 'inception_v3':
            model = get_inception_v3_model(num_classes, device_index, to_use_pre_trained_weights)
        elif model_name == 'teacher_student':
            # model = get_teacher_student_model(num_classes, device_index, mod, to_use_pre_trained_weights)
            model = get_teacher_student_model(num_classes, device_index, mod, to_use_pre_trained_weights, teacher_name, student_name, fold)
        elif model_name == 'student_teacher':
            model =  get_student_teacher_model(num_classes, device_index, mod,  to_use_pre_trained_weights)        
        elif model_name == 'teacher_teacher':
            model =  get_teacher_teacher_model(num_classes, device_index, mod,  to_use_pre_trained_weights)
        elif model_name == 'student_student':
            model =  get_student_student_model(num_classes, device_index, mod,  to_use_pre_trained_weights)
        elif model_name == 'bidirectional_kd':
            model = get_bidirectional_teacher_student_model(num_classes, device_index, mod,  to_use_pre_trained_weights)
        else:
            print("requires a model name")
    else:
        raise ValueError('Please select a model name')
    return model
