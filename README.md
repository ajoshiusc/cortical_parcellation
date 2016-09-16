# README #

This README would normally document whatever steps are necessary to get your application up and running.

### What is this repository for? ###

* Quick summary

cortical parcellation of the sub-region of the brain based on the resting state fMRI.

APPROACH: ROI list was selected and based on the correlation to the voxels within the ROI region,the region was subdivided into different number cluster.we have used spectral clustering to divide the region into different subregion.The correlation matrix was computed and was inputted to the spectral clustering algorithm.

#number of cluster:we have used two idea for selecting the number of cluster 
						1: elbow Method:based on the PCA,we will decide the number of cluster.we plotted the explained variance ratio vs number of cluster graph,point where maximum bending occurs gives us the optimal number of cluster
						2: Silhouette Method:based on how each voxel is connected to the voxels in each cluster and to the neighbouring cluster,avg silhouette score was obtained and the #nCluster which corresponds to the maximum average 							   sihouette,is the optimal number of cluster.

* Version
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### How do I get set up? ###

* Summary of set up
* Configuration
* Dependencies
* Database configuration
* How to run tests
* Deployment instructions

### Contribution guidelines ###

* Writing tests
* Code review

first select the ROI_LIST and run all_in_one_main(specify the file name at the bottom of the code) and the run slihouette(specify the name of the file) and you can select the number of cluster.
main precuneus: is the main file.#roilist : specify the ROI region, the code runs for the session 1 left to right scan over 40 subjects.we store the correlation within the ROI region,correaltion to the rest of the brain and other data.
main_file:run the file to see the graphical results and functional connectivity.

varuability_ratio: runs the code for all the session(i.e 4 session) for #ncluster

* Other guidelines

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact
