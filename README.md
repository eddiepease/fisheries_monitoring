# fisheries_monitoring
A solution to the Nature Conservancy Fisheries Monitoring Kaggle competition - https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring

###Python Scripts
* helper_code: contains various functions that are imported into the main script
* model: defining the model that is run
* read_data: reading in and normalising the data. To save time, I have saved the processed/normalised images into .npy files (located in the /data folder)
* main: the main script which trains the model by cross-val and outputs the predictions

###Folders you need to have
* Suppose your local repository is a directory called *fisheries_monitoring*.  You need to: 
    * Create a new directory *fisheries_monitoring/data*
    * Put into this directory:
        * *train* folder, containing training data, unzipped, that you download from the Kaggle website
        * *test_stg1* folder, containing test data, unzipped, that you download from the Kaggle website
    * Create a new directory *fisheries_monitoring/results*

###General information
The script to classify the pictures is in keras (with tensorflow back-end). I have tried for a couple of days to replicate the script in tensorflow but couldn't get it working. It is very close to being finished though, if you want to have a look at it, look at the 'improve_baseline' branch. It shouldn't matter if anyone else were to pick it up though as the code can be quite modular and the significant increases in accuracy can be gained by seperate modules (see below).

To install keras on Mac when you are using conda just activate whatever python environment you are using and call:
conda install -c conda-forge keras=1.2.2

This dataset is tricky in part because the fish is only a very small part of the image and several of the images are taken in quick succession one after another. Therefore, it is hard to split into a effective train/test set and get a reliable score on the Kaggle dataset. So the script that is uploaded is 'learning' the boat rather than the fish. 

###Suggested future developments
A bit more accuracy could be gained by changing optimisers/perhaps increasing the image size / making the architecture more complicated. However, these gains are likely to be minimal in comparison to the below, I suggest the following:

* Training an algorithm to draw bounding boxes / detecting the fish
* Training an algorithm to identify the mouth and tail of the fish - https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/discussion/25565

This competition is very similar to the following competition in which the winner wrote up their approach - http://blog.kaggle.com/2016/01/29/noaa-right-whale-recognition-winners-interview-1st-place-deepsense-io/
