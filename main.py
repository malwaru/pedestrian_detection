# from detector import PersonDetecor
from detector import PersonDetecor
from utilities import ImageExtractor
from svm_trainer import SvmTrainer

def test_detect():
    '''
    A test script for detectiosn
    
    '''
    test_path="./data/rawdata/leftImg8bit/test/mainz"
# file_name="berlin_000122_000019_leftImg8bit.png"
    file_name=['mainz_000001_044619_leftImg8bit.png','mainz_000001_041284_leftImg8bit.png','mainz_000001_041172_leftImg8bit.png','mainz_000000_008645_leftImg8bit.png']
    path_cell_16="./data/models/svm_model_10_01_2023_14_57_48.dat"
    
    svm_classifier=PersonDetecor(path_test=test_path,path_anno="",model_path=path_cell_16)
    svm_classifier.detect_persons_single(file_name=file_name[2],visualise=True)


def test_extract():
    '''
    
    '''
        # from jena upto end ex ept krefelf and monchenglad only negatives

    print(f"Start image extractor")
    positi_extractor=ImageExtractor(raw_path='./data/rawdata/leftImg8bit/train/zurich',anno_path='./data/annotations/gtBboxCityPersons/train/zurich',save_image=True,visualise=False)
    positi_extractor.extract_positive()


def test_trainer():
    '''
    
    '''

    trainer=SvmTrainer(file_path='./data/test_svm_data/',model_save_path='./data/models/')
    #load positives
    trainer.load_train_data(category_path='positive',data_class=1)
    #load neagtive
    trainer.load_train_data(category_path='negative',data_class=0)
    #Train data
    trainer.train_svm()





if __name__=='__main__':
    # test_extract()
    # test_trainer()
    test_detect()
   