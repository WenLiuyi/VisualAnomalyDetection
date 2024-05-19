import os
from pathlib import Path

def get_dataset_path():
    current_directory=Path.cwd()
    if current_directory.name=='anomalib_backend':
        root_directory=current_directory.parent
    else:
        root_directory=current_directory
    return root_directory

from anomalib.data.utils import DownloadInfo,download_and_extract

def download_dataset(name,url,hashsum):
    dataset_download_info=DownloadInfo(
        name=name,
        url=url,
        hashsum=hashsum
    ),

    download_and_extract(root=get_dataset_path(),info=dataset_download_info)

from anomalib.data import Folder
from anomalib import TaskType
from anomalib.data.utils import TestSplitMode


def prepare_custom_datamodule(directory_name,
    image_size,
    has_abnormal_samples=False,
    normal_dir='normal',abnormal_dir='abnormal',
    msk_dir=None,
    train_batch_size=32,eval_train_batch_size=32,
    num_workers=8,
    task=TaskType.CLASSIFICATION
    ):

    # Build Dataset
    # Condition 1:Classification task: build with normal and abnormal images
    if has_abnormal_samples and task==TaskType.CLASSIFICATION:
        datamodule=Folder(name='train_set', 
            root=get_dataset_path()/directory_name,
            normal_dir=normal_dir,
            abnormal_dir=abnormal_dir,

            train_batch_size=train_batch_size, 
            eval_batch_size=eval_train_batch_size, 
            num_workers=num_workers, 
            task=task,          #Could be: classification, detection or segmentation

            image_size=image_size, 
            transform=None, train_transform=None, eval_transform=None, 
            test_split_mode=TestSplitMode.FROM_DIR, test_split_ratio=0.2, 
            #val_split_mode=ValSplitMode.FROM_TEST, val_split_ratio=0.5, 
            seed=None)

    # Condition 2: Segmentation task: With only normal images in training set, 
        #both abnormal/normal images in test set,
        #and masks for abnormal images in test set
    elif has_abnormal_samples and task==TaskType.SEGMENTATION:
        datamodule=Folder(name='train_set',
            image_size=image_size,
            root=get_dataset_path()/directory_name,
            normal_dir=normal_dir,
            mask_dir=msk_dir,  #Path to the directory containing the mask annotations

            train_batch_size=train_batch_size, 
            eval_batch_size=eval_train_batch_size, 
            num_workers=num_workers, 
            test_split_mode=TestSplitMode.SYNTHETIC,
            task=task,
        )
    
    # Condition 3: With only normal images
        # generate synthetic abnormal images from normal images, for validation and test steps
    else:
        datamodule=Folder(name='train_set',
            root=get_dataset_path()/directory_name,
            normal_dir=normal_dir,

            train_batch_size=train_batch_size, 
            eval_batch_size=eval_train_batch_size, 
            num_workers=num_workers, 
            test_split_mode=TestSplitMode.SYNTHETIC,
            task=task,
            
            image_size=image_size
        )
    return datamodule


from anomalib.models import Padim
def prepare_model():
    model = Padim(
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"]
)
    return model


from anomalib.engine import Engine
from anomalib.utils.normalization import NormalizationMethod

def training(datamodule,model,nncf_optimization=False):
    '''if nncf_optimization:
        config["optimization"]["nncf"] = {
            "apply": True,
            "input_info": {"sample_size": [1, 3, 256, 256]},
            "compression": {
            "algorithm": "quantization",
            "preset": "mixed",
            "initializer": {"range": {"num_init_samples": 250}, "batchnorm_adaptation": {"num_bn_adaptation_samples": 250}},
        },
}
'''
    engine=Engine(
        normalization=NormalizationMethod.MIN_MAX,
        threshold="F1AdaptiveThreshold",
        task=TaskType.CLASSIFICATION,
        image_metrics=["AUROC"],
        accelerator="auto",
        check_val_every_n_epoch=1,
        devices=1,
        max_epochs=1,
        num_sanity_val_steps=0,
        val_check_interval=1.0,
    )
    engine.fit(model=model, datamodule=datamodule)
    print("hello")
    return engine

def validation(engine,model,datamodule):
    test_results=engine.test(model=model,datamodule=datamodule)
    return test_results


from anomalib.deploy import ExportType

def export_model(engine,model,export_root):
    #Exporting model to OpenVINO
    openvino_model_path=engine.export(
        model=model,
        export_type=ExportType.OPENVINO,
        export_root=export_root
    )


import timm
if __name__ == '__main__':
    #model = timm.create_model('resnet18', pretrained=True)

    datamodule=prepare_custom_datamodule('TrainSet',
        image_size=(3500,2472),
        has_abnormal_samples=True,
        normal_dir='normal',abnormal_dir='abnormal/defects',
        )

    model=prepare_model()
    training(datamodule=datamodule,model=model)
