# CompGraph
This is the code of our framework, CompGraph, which has been effectively deployed in JD Logistics for enhancing customer conversion. The following figure illustrates that, based on our prediction results, we made over 150,000 phone calls in China before August 2023. This strategy yielded an accuracy rate three times higher than the current standard practices. We are still using this framework for recommending customers with more potential value. As of December 1, 2023, more than 50,000 phone calls have been made in the last week alone. For more comprehensive details about our framework, please refer to our paper 'Complex-Path: Efficient Customer Conversion Prediction in Billion-Scale Heterogeneous Graph,' currently under review.

![Deployment](./fig/Deployment_1013.png)

# Introduction for Complex-Path
The cornerstone of our strategy lies in the definiton 'complex-path', an innovative approach designed to extract detailed information from heterogeneous graphs. Unlike the traditional meta-path technique, complex-path is adept at capturing both the attribute information and the complex structures in the heterogeneous graph. Below are the difference in the definition of meta-path and complex-path.
<p align="center">
  <img src="./fig/meta-path.png" width="50%">
  <img src="./fig/complex-path.png" width="50%">
</p>

The figures below show how complex-path effectively identifies companies related to our target under complex conditions, a task challenging for the meta-path method. Specifically, it shows how to select companies that are in the top 5 most frequently used delivery regions of the target company, share the same industry, and have signed contracts within the past 90 days.

![Meta-Path](./fig/Meta-Path_1022.png)
*Example for meta-path: it can only find the co-delivery region relaionship or co-industry relationship seperately*

![Complex-Path](./fig/CompPath_1022.png)
*Example for complex-path*

# Brief Description for Our Code
The primary function of our code is to efficiently extract information based on complex-paths from a billion-scale heterogeneous graph, utilizing the distributed system Spark. The extracted data is then used to train our model and make prediction. We are continuously refining and optimizing our code, and will keep updating it. Currently, we have achieved the capability to perform a comprehensive weekly ranking for over one billion customers who have utilized JD Logistics services.

The 'kg_utils' folder includes our methods for extracting information from heterogeneous graphs, developed using Spark and complex-path. This folder also includes baseline methods like meta-path and k-hop sampling.

The 'kg_model' folder contains our model implemented with PyTorch, along with the baseline models referenced in our paper.

The 'Jupyter' folder includes code that utilizes the functions from the aforementioned folders to implement specified features.