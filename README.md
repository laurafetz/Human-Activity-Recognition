# Human activity recognition from smartphone sensors

I studied whether smartphone accelerometer and gyroscope signals could classify twelve activities and postural transitions. I worked with a course version of a dataset from 30 volunteers aged 19–48, recorded at 50 Hz. I extracted time- and frequency-domain features from 128-sample epochs and compared four classifiers with ten-fold cross-validation. Multinomial logistic regression had the highest accuracy in the original notebook: 89.21%.

## Results

| Classifier | Best epoch-level CV accuracy |
| --- | ---: |
| Multinomial logistic regression | 89.21% |
| Scaled KNN | 87.57% |
| LDA | 86.68% |
| KNN | 74.45% |

The table comes from [original_model_comparison.csv](results/original_model_comparison.csv). I retain the original notebook and [tuning results](results/original_cv_details.csv). These are the original coursework results; I have not rerun the full comparison with the current script.

<details>
<summary>Code and files</summary>

The R script requires `tidyverse`, `caret`, `nnet`, and `e1071`.

```bash
Rscript analysis.R
```

The script reads the included training and test recordings. It writes `results/rerun_model_comparison.csv` and `results/submission_final.csv` when run. The file list below shows the files currently committed.

```text
Human-Activity-Recognition/
├── .gitattributes
├── .gitignore
├── README.md
├── analysis.R
├── data/
│   ├── README.txt
│   ├── RawData/
│   │   ├── Test/
│   │   │   ├── acc_exp03_user02.txt
│   │   │   ├── acc_exp04_user02.txt
│   │   │   ├── acc_exp07_user04.txt
│   │   │   ├── acc_exp08_user04.txt
│   │   │   ├── acc_exp17_user09.txt
│   │   │   ├── acc_exp18_user09.txt
│   │   │   ├── acc_exp19_user10.txt
│   │   │   ├── acc_exp20_user10.txt
│   │   │   ├── acc_exp21_user10.txt
│   │   │   ├── acc_exp24_user12.txt
│   │   │   ├── acc_exp25_user12.txt
│   │   │   ├── acc_exp26_user13.txt
│   │   │   ├── acc_exp27_user13.txt
│   │   │   ├── acc_exp36_user18.txt
│   │   │   ├── acc_exp37_user18.txt
│   │   │   ├── acc_exp40_user20.txt
│   │   │   ├── acc_exp41_user20.txt
│   │   │   ├── acc_exp48_user24.txt
│   │   │   ├── acc_exp49_user24.txt
│   │   │   ├── gyro_exp03_user02.txt
│   │   │   ├── gyro_exp04_user02.txt
│   │   │   ├── gyro_exp07_user04.txt
│   │   │   ├── gyro_exp08_user04.txt
│   │   │   ├── gyro_exp17_user09.txt
│   │   │   ├── gyro_exp18_user09.txt
│   │   │   ├── gyro_exp19_user10.txt
│   │   │   ├── gyro_exp20_user10.txt
│   │   │   ├── gyro_exp21_user10.txt
│   │   │   ├── gyro_exp24_user12.txt
│   │   │   ├── gyro_exp25_user12.txt
│   │   │   ├── gyro_exp26_user13.txt
│   │   │   ├── gyro_exp27_user13.txt
│   │   │   ├── gyro_exp36_user18.txt
│   │   │   ├── gyro_exp37_user18.txt
│   │   │   ├── gyro_exp40_user20.txt
│   │   │   ├── gyro_exp41_user20.txt
│   │   │   ├── gyro_exp48_user24.txt
│   │   │   └── gyro_exp49_user24.txt
│   │   └── Train/
│   │       ├── acc_exp01_user01.txt
│   │       ├── acc_exp02_user01.txt
│   │       ├── acc_exp05_user03.txt
│   │       ├── acc_exp06_user03.txt
│   │       ├── acc_exp09_user05.txt
│   │       ├── acc_exp10_user05.txt
│   │       ├── acc_exp11_user06.txt
│   │       ├── acc_exp12_user06.txt
│   │       ├── acc_exp13_user07.txt
│   │       ├── acc_exp14_user07.txt
│   │       ├── acc_exp15_user08.txt
│   │       ├── acc_exp16_user08.txt
│   │       ├── acc_exp22_user11.txt
│   │       ├── acc_exp23_user11.txt
│   │       ├── acc_exp28_user14.txt
│   │       ├── acc_exp29_user14.txt
│   │       ├── acc_exp30_user15.txt
│   │       ├── acc_exp31_user15.txt
│   │       ├── acc_exp32_user16.txt
│   │       ├── acc_exp33_user16.txt
│   │       ├── acc_exp34_user17.txt
│   │       ├── acc_exp35_user17.txt
│   │       ├── acc_exp38_user19.txt
│   │       ├── acc_exp39_user19.txt
│   │       ├── acc_exp42_user21.txt
│   │       ├── acc_exp43_user21.txt
│   │       ├── acc_exp44_user22.txt
│   │       ├── acc_exp45_user22.txt
│   │       ├── acc_exp46_user23.txt
│   │       ├── acc_exp47_user23.txt
│   │       ├── acc_exp50_user25.txt
│   │       ├── acc_exp51_user25.txt
│   │       ├── acc_exp52_user26.txt
│   │       ├── acc_exp53_user26.txt
│   │       ├── acc_exp54_user27.txt
│   │       ├── acc_exp55_user27.txt
│   │       ├── acc_exp56_user28.txt
│   │       ├── acc_exp57_user28.txt
│   │       ├── acc_exp58_user29.txt
│   │       ├── acc_exp59_user29.txt
│   │       ├── acc_exp60_user30.txt
│   │       ├── acc_exp61_user30.txt
│   │       ├── gyro_exp01_user01.txt
│   │       ├── gyro_exp02_user01.txt
│   │       ├── gyro_exp05_user03.txt
│   │       ├── gyro_exp06_user03.txt
│   │       ├── gyro_exp09_user05.txt
│   │       ├── gyro_exp10_user05.txt
│   │       ├── gyro_exp11_user06.txt
│   │       ├── gyro_exp12_user06.txt
│   │       ├── gyro_exp13_user07.txt
│   │       ├── gyro_exp14_user07.txt
│   │       ├── gyro_exp15_user08.txt
│   │       ├── gyro_exp16_user08.txt
│   │       ├── gyro_exp22_user11.txt
│   │       ├── gyro_exp23_user11.txt
│   │       ├── gyro_exp28_user14.txt
│   │       ├── gyro_exp29_user14.txt
│   │       ├── gyro_exp30_user15.txt
│   │       ├── gyro_exp31_user15.txt
│   │       ├── gyro_exp32_user16.txt
│   │       ├── gyro_exp33_user16.txt
│   │       ├── gyro_exp34_user17.txt
│   │       ├── gyro_exp35_user17.txt
│   │       ├── gyro_exp38_user19.txt
│   │       ├── gyro_exp39_user19.txt
│   │       ├── gyro_exp42_user21.txt
│   │       ├── gyro_exp43_user21.txt
│   │       ├── gyro_exp44_user22.txt
│   │       ├── gyro_exp45_user22.txt
│   │       ├── gyro_exp46_user23.txt
│   │       ├── gyro_exp47_user23.txt
│   │       ├── gyro_exp50_user25.txt
│   │       ├── gyro_exp51_user25.txt
│   │       ├── gyro_exp52_user26.txt
│   │       ├── gyro_exp53_user26.txt
│   │       ├── gyro_exp54_user27.txt
│   │       ├── gyro_exp55_user27.txt
│   │       ├── gyro_exp56_user28.txt
│   │       ├── gyro_exp57_user28.txt
│   │       ├── gyro_exp58_user29.txt
│   │       ├── gyro_exp59_user29.txt
│   │       ├── gyro_exp60_user30.txt
│   │       ├── gyro_exp61_user30.txt
│   │       └── labels_train.txt
│   ├── activity_labels.txt
│   └── example_submission.csv
├── notebook/
│   └── Human_Activity_Recognition_Notebook.ipynb
└── results/
    ├── original_cv_details.csv
    └── original_model_comparison.csv
```

</details>

## Limitations

Epochs from the same person can enter different folds, so the reported accuracy does not estimate performance on unseen people. I filtered predictors before cross-validation, which may inflate the estimates. The withheld test labels prevent me from reporting test accuracy.

## Credits

I worked on this project with **Yuxuan Xiang**. Project authors: **Laura Maria Fetz** and **Yuxuan Xiang**. We shared feature extraction and model selection; I wrote the notebook text and prepared its layout and figures. The dataset documentation in [data/README.txt](data/README.txt) credits Jorge L. Reyes-Ortiz, Davide Anguita, Luca Oneto, Xavier Parra, and collaborators.
