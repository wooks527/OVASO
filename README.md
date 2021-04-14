OVASO
======

This is a repository for OVASO model which classify COVID-19 from chest X-ray images. The proposed model is simple and competitive with COVID-Net which is a state-of-the-art model among COVID-19 Classification models using chest X-ray images.

## Performance

| Model | Accuracy || Recall ||| Precision ||
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| - | - | COVID-19 | Pneumonia | Normal | COVID-19 | Pneumonia | Normal |
| COVID-Net CXR4-A | 94.3% | 95% | 94% | 94% | 99% | 93.1% | 91.3% |
| COVID-Net CXR4-B | 93.7% | 93% | 92% | 96% | 98.9% | 93.9% | 88.9% |
| COVID-Net CXR4-C | 93.3% | 96% | 89% | 95% | 96% | 93.7% | 90.5% |
| OVASO | 95.3% | 93% | 96% | 97% | 98.9% | 96.0% | 91.5% |

## Tutorials

Using below ipyhon file, you can evaluate OVASO model.

* Evaluate the model
  - [evaluate_the_model.ipynb](evaluate_the_model.ipynb)
