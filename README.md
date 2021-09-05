# ImgRec
This read me file describes the steps to run the code for "End-to-End Image-Based Recommender System" paper.

Requirements*
1- numpy
2- pandas
3- tensorflow 2.3.0 
4- scikit-learn

To run any of the scripts it is required to check the "data path" of the required file, then
running command is ==> python file_name e.g. "python imgrec_amazon_fashion_ResNetFeatures_FT.py"

For ImgRec-EtE, we split the training process into two phases to save time and processing power;
Phase 1: fixing the image network parameters.
Phase 2: jointly learning the last 50 layers of the image network.

To run the experimemts firstly run the "imgrec_men_finetuned_SaveModel.py", which will run with fixed pre-trained parameters.
Then run the "imgrec_men_finetuned_LoadModel.py" to obtain the paper's results.

