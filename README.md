# 3D_TransPOSE
A Study on 3D Pose Estimation in Invisible Area Using Radar Signal and Machine Learning


## Pretrained Model

https://drive.google.com/file/d/1sZYllTCnyNS4_KU7zE2YVqvMmFCVBPgP/view?usp=sharing

    python main.py --resume_checkpoint ./outputs/checkpoint.pth
  
## Dataset

dataset download

https://drive.google.com/file/d/1Fkp_6UoefdByJaonBbvlaQ9aoDNYcf7l/view?usp=sharing

    📂train
    ├ 📂radar
    │  ├ 📂A
    │  │  ├ 📂one
    │  │  │  ├ session_1
    │  │  │  ├ session_2
    │  │  │  └    ⁝
    │  │  └ 📂two
    │  │     ├ session_1
    │  │     └    ⁝
    │  ├ 📂B
    │  │  ├ 📂one
    │  │  └ 📂two
    │  ├ 📂C
    │  │  ├ 📂one
    │  │  └ 📂two
    │  └ 📂D
    │   
    └ 📂motive
       ├ 📂A
       │  ├ 📂one
       │  │  ├ session_1
       │  │  ├ session_2
       │  │  └    ⁝
       │  └ 📂two
       │     ├ session_1
       │     └    ⁝
       ├ 📂B
       │  ├ 📂one
       │  └ 📂two
       ├ 📂C
       │  ├ 📂one
       │  └ 📂two
       └ 📂D

    same as 📂test dataset folder  
