# Dogs in portraits of the 16th-18th centuries: a children's companion or a symbol of their majesty?
This repository is dedicated for our course project of DH-412: History and the digital.


### Structure  
├── data/                                                       // data  
│   ├──kaggle_painters_by_numbers                               //   
│   │   └── all_data_info.csv                                   // matadata of all the paintings in the dataset    
│   └────── dog_paintings.csv                                   // matadata of portraits containing dog in the dataset    
├── report/                                                     // project report  
├── scripts                                                     //   
│   ├──s0_dataset_exploration                                   //   
│   │   ├── plot_of_statistics.ipynb                            // plot of statistics  
│   │   ├── images_exploration.ipynb                            // exploration of images attributes  
│   │   └── dataset_exploration.ipynb                           // exploration of metadata  
│   ├──s1_model_experimentation                                 //  
│   │   ├── displaying_paintings_for_every_50_years.ipynb       // show the paintings of portraits with dog   
│   │   ├── loading_error_demo.ipynb                            // loading error  
│   │   ├── object_detection_original.ipynb                     // object detection  
│   │   ├── s0_kaggle_object_detection.ipynb                    // kaggle object detection  
│   │   ├── s1_objects_statistics.ipynb                         // statistics of objects  
│   │   ├── s2_association_rules_mining.ipynb                   // association rules mining  
│   │   └── utils.py                                            //   
│   ├──s2_dogs_analysis                                         //  
│   │   ├── s2_association_rules_mining.ipynb                   // new association rules mining  
│   │   └── s3_typologies.ipynb                                 // typologies  
│   ├──data_filepaths.py                                        // path for running the object detection  
│   ├──plot_styles.py                                           // plot styles  
│   └── __init__.py                                             //  
├── src/                                                        //  
│    └── __init__.py                                            //  
├── LICENSE                                                     // License  
├── README.md                                                   // Readme file  
└── conda_req.yml                                               // package requirement  

## Dataset
The dataset used in this project is obtained from a Kaggle competition “Painters by numbers”. It contains 103'250 paintings of different artists. The Kaggle competition focused on being able to identify whether two paintings were from the same artist. More than 95\% of the images were obtained from WikiArt. For the aim of our project, we focus on portraits and self-portraits, totalling 18'378 paintings. The resulting dataset contains works of 1'020 different artists in 94 different painting styles. Most paintings are from the 19th and early 20th century, while 3'134 paintings were created before 1800. 

## Research questions

Why and in which circumstances were dogs and children represented together in the portrait paintings of the 16th-18th centuries? 

What similarities can we identify in some of these paintings? 

Could these representations of dogs communicate any additional information about the depicted children?




## 👤 Contributors
Didier Dupertuis, Yuxiao Li, Irina Serenko
