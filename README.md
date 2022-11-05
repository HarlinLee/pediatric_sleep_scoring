[Automatic Sleep Scoring from Large-scale Multi-channel Pediatric EEG](https://arxiv.org/pdf/2207.06921.pdf)
---
by Harlin Lee, Aaqib Saeed

### Abstract
Sleep is particularly important to the health of infants, children, and adolescents, and sleep scoring is the first step to accurate diagnosis and treatment of potentially life-threatening conditions. But pediatric sleep is severely under-researched compared to adult sleep in the context of machine learning for health, and sleep scoring algorithms developed for adults usually perform poorly on infants. Here, we present the first automated sleep scoring results on a recent large-scale pediatric sleep study dataset that was collected during standard clinical care. We develop a transformer-based model that learns to classify five sleep stages from millions of multi-channel electroencephalogram (EEG) sleep epochs with 78% overall accuracy. Further, we conduct an in-depth analysis of the model performance based on patient demographics and EEG channels. The results point to the growing need for machine learning research on pediatric sleep.

#### Installation
Install required packages as follows:
```
pip3 install -r requirements.txt
```

#### Dataset 
Get access to [NCH Sleep DataBank](https://sleepdata.org/datasets/nchsdb).

Preprocess and convert the data to TFRecords format or see an example [here](https://github.com/liboyue/sleep_study) to use Pytorch.

We provide train, validation and test splits in the `splits` directory. The `file_path` has following structure `<study_pat_id>_<sleep_study_id>_<eeg_example_index>_<age_group>.<ext>`, where `<ext>` can be ignored. `pat_id` is patient id, and `age_group` represents age group the patient belongs to.

#### Training: 
```
python3 main.py
```

### Citation
```
@inproceedings{Lee2022Automatic,
  title={Automatic Sleep Scoring from Large-scale Multi-channel Pediatric EEG},
  author={Harlin Lee and Aaqib Saeed},
  journal={arXiv preprint arXiv:2207.06921},
  year={2022}
}
```
