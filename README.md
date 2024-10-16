# Targeted Visual Prompting for Medical Visual Question Answering

Official implementation of the Paper "Targeted Visual Prompting for Medical Visual Question Answering," presented at the AMAI2024 Workshop of the MICCAI 2024 Conference. For more details, please refer to our [paper](https://arxiv.org/pdf/2408.03043).

**This repo is undergoing a cleaning and organization process.**

## Installing Requirements
After cloning the repo, create a new environment, activate it, and then install the required packages by running:

    pip install -r requirements.txt

## Data
Download the original data from [here](https://zenodo.org/records/8192556) and the processed annotation files from [here](https://drive.google.com/file/d/1IajA4uwbXXY6J-S6w4tbAIVitnWvx0vl/view?usp=sharing). Alternatively, run the `prepare_data.py` under the folder corresponding to each dataset (ris, insegcat or dme) to prepare the data.

**Depending on where you place the downloaded data, you will need to configure the paths in the subsequent steps.**

## Finetuning
To run the code, use the bash scripts in the folder `scripts_vqa`, for example, to run the baseline `crop region` on the RIS dataset,

    bash scripts_vqa/ris/crop_region.sh

Notice the paths to the datasets have to be changed in advance.

## Testing

The test scripts are located in the same folders as for fine-tuning. The same command will be effective to evaluate performance. For example, to evaluate the baseline `draw region` on the DME dataset, use

    bash scripts_vqa/dme/draw_region_test.sh

Similar to the finetuning, the paths have to be configured in advance.

The metrics will be printed automatically at the end of the inference process.


<br />
<br />

This work was carried out at the [AIMI Lab](https://www.artorg.unibe.ch/research/aimi/index_eng.html) of the [ARTORG Center for Biomedical Engineering Research](https://www.artorg.unibe.ch) of the [University of Bern](https://www.unibe.ch/index_eng.html). Please cite this work as:

> @article{tascon2024targeted,\
  title={Targeted Visual Prompting for Medical Visual Question Answering},\
  author={Tascon-Morales, Sergio and M{\'a}rquez-Neila, Pablo and Sznitman, Raphael},\
  journal={arXiv preprint arXiv:2408.03043},\
  year={2024}\
}
