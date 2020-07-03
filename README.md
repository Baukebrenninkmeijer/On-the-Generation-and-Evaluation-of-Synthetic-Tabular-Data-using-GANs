# On the Generation and Evaluation of Synthetic Tabular Data using GANs

## Overview
- Master Thesis Data Science, Radboud University 2019
- License: MIT
- Based on the awesome work from the guys at MIT Data to AI Lab. ([TGAN](https://github.com/DAI-Lab/TGAN), [SDGym](https://github.com/DAI-Lab/SDGym))

## Abstract
With privacy regulations becoming stricter, the opportunity to apply synthetic data is growing rapidly. Synthetic data can be used in any setting where access to data with personal information is not strictly necessary. However, many require the synthetic data to present the same relations as the original data. Existing statistical models and anonymization tools often have adverse effects on the quality of data for downstream tasks like classification. Deep learning based synthesization techniques like GANs provide solutions for cases where it is vital these relations are kept. Inspired by GANs, we propose an improvement in the state-of-the-art in maintaining these relations in synthetic data. Our proposal includes three contributions. First, we propose the addition of skip connections in the generator, which increases gradient flow and modeling capacity. Second, we propose using the WGAN-GP architecture for training the GAN, which suffers less from mode-collapse and has a more meaningful loss. And finally, we propose a new similarity metric for evaluating synthetic data. This metric better captures different aspects of synthetic data when comparing it to real data. We study the behaviour of our proposed model adaptations against several baseline models on three datasets. Our results show that our proposals improve on the state-of-the-art models, by creating higher quality data. Our evaluation metric captures quality improvements in synthetic data and gives detailed insight into the strengths and weaknesses of evaluated models. We conclude that our proposed adaptations should be used for data synthesis, and our evaluation metric is precise and gives a balanced view of different aspects of the data. 

The data evaluation library can be found in an additional repository: [https://github.com/Baukebrenninkmeijer/Table_Evaluator](https://github.com/Baukebrenninkmeijer/Table_Evaluator).

## Motivation
To see the motivation for my decisions, please have a look at my master thesis, found at https://www.ru.nl/publish/pages/769526/z04_master_thesis_brenninkmeijer.pdf



## Using this work?
If you're using this work, please cite the following work:

```
@article{brenninkmeijer2019synthetic,
  title={On the Generation and Evaluation of Synthetic Tabular Data using GANs},
  author={Bauke Brenninkmeijer, Youri Hille, Arjen P. de Vries},
  year={2019}
}
```
