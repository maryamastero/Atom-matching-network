## Learning symmetry-aware atom mapping in chemical reactions through deep graph matching
This repository contains the data and codes of the paper: 

[Learning symmetry-aware atom mapping in chemical reactions through deep graph matching]([Learning symmetry-aware atom mapping in chemical reactions through deep graph matching](https://link-url-here.org](https://eur01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fdoi.org%2F10.1186%2Fs13321-024-00841-0&data=05%7C02%7Cmaryam.astero%40aalto.fi%7C36e9a4806531421ecd8a08dc630e71ee%7Cae1a772440414462a6dc538cb199707e%7C1%7C0%7C638494162762597865%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C0%7C%7C%7C&sdata=uyfgOpXamRvbwP25OVNa%2BXR3Kl2SBslkRyxLDUEIFJM%3D&reserved=0))

![amnet](https://github.com/maryamastero/Atom-matching-network/assets/60658276/595e55c1-014f-428a-a177-e31fd760ba3c)


 # Model

The initial step involves transforming molecular structures into graphs, incorporating atom and bond features that encapsulate their distinctive attributes. The molecular graph is then processed by graph isomorphism networks (GIN). This process allows each node within the input molecular graph to be transformed into an embedding space. These node embeddings capture both the topological structure of the nodes and their features. To achieve this embedding, a shared weight neural network takes as input the adjacency matrices of both molecular graphs, as well as their node feature and edge features. This process brings both molecular graphs into the same space; therefore, pairwise matching scores can be computed between the nodes of ${G}_R$ and ${G}_P$ using a similarity function (e.g., dot product), which takes as input the features of two vectors, and its output is a scalar similarity score. These pairwise matching scores are stored in the initial correspondence matrix.

Then, to obtain the pairwise matching probabilities, we normalize the matrix $\hat{M}$ row-wise. The matrix can be interpreted as a correspondence matrix that assigns a probability to each pair of nodes in ${G}_R$ and ${G}_P$, indicating the likelihood of each node in ${G}_R$ being matched with each node in ${G}_P$.

After that, to not penalize the model for failing to distinguish between chemically equivalent atoms, we take advantage of molecular symmetry information. This approach recognizes the inherent symmetry and allows the model to focus on distinguishing between non-equivalent atoms, resulting in a more efficient and accurate atom mapping process.

To evaluate the effectiveness of models, we report the percentages of correctly mapped reactions at the top@1, top@3, and top@5 and the average accuracy of the prediction on the test dataset. Top@k indicates the number of reactions correctly mapped when the mapped atom is correct in the first top prediction. The average Accuracy of atom mapping is calculated by summing up the accuracy of the predicted atom mapping of each reaction and then dividing it by the total number of reactions in the test set. The performance of our method is evaluated across a number of tasks, each contributing to a comprehensive assessment.

In our initial task, our primary objective was to evaluate the effect of identifying molecular symmetry on atom mapping predictions. This experiment involves comparing models that incorporate the identification of molecular symmetry with those that do not. 

Our second task explores understanding the influence of feature selection on the performance of the AMNet. This step is pivotal in understanding how the choice of features impacts the accuracy and overall quality of our atom mapping predictions. 

In our final task, we evaluate our model's performance using a subset of the widely recognized Golden dataset. This dataset is commonly used in the evaluation of various atom mapping methods. Our assessment against this benchmark dataset provides valuable insights into how our model's capabilities compare to other established techniques.


 # Dataset
 The dataset was sourced from https://github.com/wengong-jin/nips17-rexgen/tree/master.

In order to evaluate how the choice of features influences the accuracy of atom mapping predictions, we conducted an analysis involving various choices of node and edge features.
Atom features

![Screenshot 2024-01-11 at 9 08 17](https://github.com/maryamastero/Atom-matching-network/assets/60658276/1c9db5df-0f23-445e-ab98-12b6b12de741)


# Identifying chemically equivalent atoms with Weisfeiler-Lehman test
Chemically equivalent atoms are atoms within a molecule that have the same chemical environment and exhibit identical properties in a given chemical context. We utilize an adaptation of the Weisfeiler-Lehman (WL) test for identifying chemically equivalent atoms within a molecule. We consider two atoms to be chemically equivalent if they have the same atomic symbol and their three hop neighbors are the same.

![Screenshot 2024-01-11 at 9 20 07](https://github.com/maryamastero/Atom-matching-network/assets/60658276/9d3ac4d0-7689-428f-a374-1b31b829184a)

 # Results
 Performance of the AMNet with and without molecule symmetry identification
 
 ![Screenshot 2024-01-11 at 9 49 50](https://github.com/maryamastero/Atom-matching-network/assets/60658276/f89586d1-79c1-448f-af96-0cc9d0191a00)


Performance of the AMNet using various choices of features on the USPTO-15k test set
![Screenshot 2024-01-11 at 9 48 34](https://github.com/maryamastero/Atom-matching-network/assets/60658276/be7d600f-b32e-4b5e-8d2a-b9468dd76803)

# Environment
- `mol_graph.yml`: Configuration file for the required environment.


# Code Structure

- `/dataset/preprocessing.py`: Preprocesses the data, removing duplicate and invalid molecules.
- `/dataset/molgraphdataset.py`: Creates a PyTorch Geometric dataset from the preprocessed data.
- `/AMNet/amnet.py`: Implementation of the Atom Mapping Network using Graph Isomorphism Network(GIN).
- `/utils/wl.py`: Weisfeiler-Lehman test for identification of equivalent atoms.
- `/train/amnet_train.py`: Training script for the AMNet model on the USPTO-15k dataset.



# Example

![Screenshot 2024-01-11 at 9 46 45](https://github.com/maryamastero/Atom-matching-network/assets/60658276/e75dc85d-61a3-4035-b3c8-207d1f4c4886)
