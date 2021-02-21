# AnimalActiveLearning
## Data
All raw camera trap images that were used in this study, along with the associated
annotation information, will be uploaded to the publicly-available Labeled Information
Library of Alexandria: Biology and Conservation (LILA BC) upon publication.

## Stage 1 pre-training
```
python main.py --config ./configs/Stage_1/plain_resnet_MOZ_S1_101920.yaml
```
## Stage 1 energy fine-tuning
```
python main.py --config ./configs/Stage_1/energy_resnet_MOZ_S1_101920.yaml --energy_ft
```
## Stage 2 training
```
python main.py --config ./configs/Stage_2/pslabel_oltr_resnet_MOZ_S2_111120.yaml
```
## Stage 2 energy fine-tuning
```
python main.py --config ./configs/Stage_2/pslabel_oltr_energy_resnet_MOZ_S2_111620.yaml
```