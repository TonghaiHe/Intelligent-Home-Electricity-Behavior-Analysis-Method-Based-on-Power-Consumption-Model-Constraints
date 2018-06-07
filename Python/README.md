### Operation method
* Put main_file.py and run_file.py in the same working directory.
* Modify the correct data set work path in the mainfile.py file. The modification is as follows,
```Python
file_path = "smart_grid_dataset.csv" #line 220
```
```Python
data=pd.read_csv("smart_grid_dataset.csv",header=0,error_bad_lines=False) #line 252
```
* Set parameters in the run_file.py file and run the file.
### Program details
#### run_file.py
variable hh - parameter Eh setting

variable ll - parameter El setting

alpha - α the parameters of iteration 1 in the iterative formula

beta - β the parameters of iteration 2 in the  iterative formula

gamma - γ the parameters of iteration 3 in the iterative formula

lanmda - λ the parameters of iteration 1 in the iterative formula

n - The dimension of matrix decomposition

```Python
for i in range(10):  #i is the repetition of the experiment
    alpha=0.01 
    beta=0.01 
    gamma=0.01 
    lanmda=0.01 
    n=70 
    for j in cc:
        h_l=j
        run(alpha,beta,gamma,lanmda,n,h_l)
    i+=1
```

#### main_file.py
##### Function introduction
```
def read_csv(path):#Read the data file, the file format is CSV.
```
```
def produce_train_data(raw_data, category):#It generates training data, input raw data and electrical category strings, and returns the matrix consisting of about 70% data, including total data.
```
```
def produce_test_data(raw_data, category):#It generates test data, enters raw data and electrical category strings, and returns the matrix consisting of about 100% data, including total data.
```
```
def cos(vector1,vector2): #Computing the cosine similarity of two vectors
```
```
def multipl(a,b):#Calculating the product of two vectors
```
```
def corrcoef(x,y):#Calculating the Pearson correlation coefficient of two vectors
```
```
def Initialization(G, X, n, k):#Initialization matrix G and X, X in the parameter is the training matrix, that is, the separate electrical matrix, G is the total data matrix, n is the decomposition of the dimension of the electrical matrix, and K is the number of matrices.The function returns to A, B, W and H. H is an approximate diagonal matrix.
```
```
def train_model_withlaplace(A, B, W, H, G, X, Z, E, n, k, alpha, beta, gamma, lanmda):#Matrix iterative decomposition function
```
```
def prediction(B, W, k):#Matrix reconfiguration function
```
```
def performance2(G, new_G, k):  #The decomposition performance evaluation function is introduced in detail in the related papers.
```
```
def run(alpha,beta,gamma,lanmda,n,h_l): #Program entry. We can run the entire program by running this function.
```
##### Variable introduction
 X = [0] * len(category)#List of data matrix for individual electrical appliances
 
 G = [0] * len(category)#Total power data matrix
 
 Z = [0] * len(category)#List. Similarity matrix between users of different electrical appliances
 
 E = [0] * len(category)#List. Homogeneity parameters between users of different electrical appliances
 
 k #Number of household appliances
 Acc2 #Evaluation results
 
### Tips
In the program code, we have detailed notes for each function and variable. If you want to know more about the operation principle of the program, please look at the source code.
