## Learning symmetry-aware atom mapping in chemical reactions through deep graph matching
This repository contains the data and codes of the paper: Learning symmetry-aware atom mapping in chemical reactions through deep graph matching

![amnet](https://github.com/maryamastero/Atom-matching-network/assets/60658276/595e55c1-014f-428a-a177-e31fd760ba3c)

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
![Screenshot 2024-01-10 at 16 16 34](https://github.com/maryamastero/Atom-matching-network/assets/60658276/c4bbb388-545b-45d8-9559-1c054374c28d)

 # Model

 
```
code test
```
# Identifying chemically equivalent atoms with Weisfeiler-Lehman test
Chemically equivalent atoms are atoms within a molecule that have the same chemical environment and exhibit identical properties in a given chemical context. We utilize an adaptation of the Weisfeiler-Lehman (WL) test for identifying chemically equivalent atoms within a molecule. We consider two atoms to be chemically equivalent if they have the same atomic symbol and their three hop neighbors are the same.
![eqa](https://github.com/maryamastero/Atom-matching-network/assets/60658276/f277516c-549c-42fd-9670-5d20635f122c)

 # Results
 Performance of the AMNet with and without molecule symmetry identification
![Screenshot 2024-01-10 at 16 07 15](https://github.com/maryamastero/Atom-matching-network/assets/60658276/d30d857a-658a-4bb8-9a55-42e3abbcb699)

Performance of the AMNet using various choices of features on USPTO-15k test set
![Screenshot 2024-01-10 at 16 07 46](https://github.com/maryamastero/Atom-matching-network/assets/60658276/e5a2d634-3bb0-4ca4-b9d3-1c7f23ef4037)
