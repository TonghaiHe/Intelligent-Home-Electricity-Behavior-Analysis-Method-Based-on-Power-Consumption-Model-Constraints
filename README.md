### Intelligent Home Electricity Behavior Analysis Method
#### Intelligent Home Electricity Behavior Analysis Method Based on Power Consumption Model Constraints
we propose an efficient and usable method for energy decomposition by exploiting the similarity between families on the basis of nonnegative matrix decomposition. This method is trained by the model of intelligent home data which has detailed electrical information. In the condition of obtaining the total power of the new user, the intelligent household electricity mode is obtained. Experiments on more than more than 200 families of datasets show that this approach does not require additional information and achieves more robust and efficient performance than existing methods. Aggregate power consumption data required by our approach is easier to obtain because of the low resolution requirements, as a result, it can be applied immediately to many families around the world. 
#### Project Members
* Zhen Yang
* He Tonghai
* Lu Ruirui
* Li Yide
#### Background
With the continuous deepening of information technology, the Internet of Things and intellisense devices have gradually penetrated into people’s daily lives and have continuously changed people’s lifestyles and behaviors. Smart Grid is one of the most successful applications of IoT technology. How to use smart grid technology to better improve people’s quality of life has gradually aroused the attention of academia and industry. Unfortunately, although the smart grid technology has achieved great success in many fields such as power dispatching and fault detection, how to use the power model information contained in the smart grid data to further tap into the information based on the user’s individual power consumption behavior in order to realize privacy protection, smart recommendation, intelligent security and other functions are still
important problems to be solved. 
#### The Introduction of the Principle of the Algorithm
Here's the brief introduction of the principle of the algorithm, it can help you understand and use the dlls and software better. Inspired by the work of J. Zico Kolter et al, we propose a novel approach that produces an energy breakdown in homes. The algorithmic approach we present builds upon nonnegative matrix factorization methods that is widely applied in image analysis, text clustering, speech processing and other fields. We consider all the factors and establish a unified decomposition model. Instead, we find that there is a similar connection between different consumers in the mode of power consumption by analyzing the existing data. For example, households with similar total power consumption are more likely to have similar thermal energy consumption as thermal energy consumption accounts for the largest proportion of households in many countries. This has also been confirmed by analyzing dataset. Based on this conclusion , we use the regularization method to add a similar relation matrix called Laplacian matrix to the proposed method for improving the accuracy of energy decomposition.

The basic schematic of this algorithm is shown below.
![image](https://github.com/TonghaiHe/Intelligent-Home-Electricity-Behavior-Analysis-Method-Based-on-Power-Consumption-Model-Constraints/blob/master/Picture/pic1.png)
The entire decomposition process of the algorithm is divided into two steps. 

The first step is the matrix coordination training module. According to the target formula, the training data matrix $X_i$ is decomposed into sub-matrix Bi and Ai, the test matrix G is decomposed into Bi and Wi in the same way. It should be noted that the first two decompositions are performed synchronously. Where i = 1,…,k; indicates different types of appliances, Bi in the two decompositions is the same matrix. 

The second step, the matrix reconstruction decomposition module. According to the target formula New_Gi = BiWi using decomposition sub-matrix Bi and Wi
reconstruct matrix. Assuming that the new matrix has M columns, the first N columns in New_Gi are the reconstructed matrix Xi, then M - N is listed as a separate electrical data decomposed from total electricity data.


