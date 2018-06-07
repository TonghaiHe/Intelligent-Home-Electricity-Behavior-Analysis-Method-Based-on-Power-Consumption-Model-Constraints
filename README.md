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

![pic1](https://github.com/TonghaiHe/Intelligent-Home-Electricity-Behavior-Analysis-Method-Based-on-Power-Consumption-Model-Constraints/blob/master/Picture/pic1.png)

The entire decomposition process of the algorithm is divided into two steps. 

The first step is the matrix coordination training module. According to the target formula, the training data matrix Xi is decomposed into sub-matrix Bi and Ai, the test matrix G is decomposed into Bi and Wi in the same way. It should be noted that the first two decompositions are performed synchronously. Where i = 1,…,k indicates different types of appliances, Bi in the two decompositions is the same matrix. 

The second step, the matrix reconstruction decomposition module. According to the target formula New_Gi = BiWi using decomposition sub-matrix Bi and Wi reconstruct matrix. Assuming that the new matrix has M columns, the first N columns in New_Gi are the reconstructed matrix Xi, then M - N is listed as a separate electrical data decomposed from total electricity data.

The optimization model after constrained by homogeneity is as follows.

![pic2](https://github.com/TonghaiHe/Intelligent-Home-Electricity-Behavior-Analysis-Method-Based-on-Power-Consumption-Model-Constraints/blob/master/Picture/pic2.png)

In the above formula, the matrix Xi (i=1, 2, 3...k) represents the training data, which is the total power consumption data of different single appliances. For example, X1 represents the refrigerator, X2 represents the air conditioner, and the matrix G represents the test data. Each column of the data matrix G represents An electricity user, each row of the data matrix G represents the power consumption of different users at the same time, and the same user's different time interval is 1 hour. Matrix L=D−Z, which is a Laplacian matrix and D is a diagonal matrix, where diagonal elements D(i,i) is the sum of the matrix Z in the line or column of the i . Z is a homogeneous coefficient matrix of n training examples defined as.

![pic3](https://github.com/TonghaiHe/Intelligent-Home-Electricity-Behavior-Analysis-Method-Based-on-Power-Consumption-Model-Constraints/blob/master/Picture/pic3.png)

The ε in the matrix Z is used to measure the similarity between users. Its definition is as follows.

![pic4](https://github.com/TonghaiHe/Intelligent-Home-Electricity-Behavior-Analysis-Method-Based-on-Power-Consumption-Model-Constraints/blob/master/Picture/pic4.png)

The matrix H is an extension of the diagonal matrix and is used to take the first N columns of the matrix to maintain the similarity of overlapping user data. Among them, α β γ represents the adjustment parameters of the degree of constraint on the regular terms, which together control the robustness of the model. λ is the adjustment parameter used to control the strength of the constraint. Ei is a coefficient matrix of homogeneity regular terms, which is used to control the strength of the regular terms of different appliances.

We use the general solution of NMF to solve the optimization function. To ensure that all the formulae in the decomposition process are positive, we use the Karush-Kuhn-Tucker (KKT) condition. The following iteration formula is obtained.

![pic5](https://github.com/TonghaiHe/Intelligent-Home-Electricity-Behavior-Analysis-Method-Based-on-Power-Consumption-Model-Constraints/blob/master/Picture/pic5.png)
![pic6](https://github.com/TonghaiHe/Intelligent-Home-Electricity-Behavior-Analysis-Method-Based-on-Power-Consumption-Model-Constraints/blob/master/Picture/pic6.png)

If you want to learn more details of this algorithm, please click here [http://********](http://www.baidu.com) to view the full paper.
#### Highlights
* We proposed a novel method for data decomposition of smart grids, and unified modeling of the methods, given the algorithm flow and operation block diagram, so that readers can easily understand the method, and quickly put the method into use.
* We fully analyze the interrelationships among the data, discover the problems of consistency and homogeneity among the data, and join the model through regular terms to improve the accuracy of the algorithm.
* Our method is suitable for ordinary homes with low-frequency sampling meters
#### Publication
Yang Zhen, He Tonghai. A data decomposition method for smart grid based on non-negative matrix factorization. National invention patent
Patent number: 201810052322.9.

Yang Zhen, He Tonghai. Research on smart home electricity consumption behavior analysis method based on power consumption constraints[D]. Beijing University of Technology, 2018.

#### Please Note
If you use the software or dynamic link library (dll) in your program or research, please indicate that the part of paper and program cites the following paper.

Yang Zhen, He Tonghai. Research on smart home electricity consumption behavior analysis method based on power consumption constraints[D]. Beijing University of Technology, 2018.
#### Code & Toolbox
[Github page]()

