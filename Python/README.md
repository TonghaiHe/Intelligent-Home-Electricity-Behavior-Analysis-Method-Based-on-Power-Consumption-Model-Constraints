### Operation method
* Put main_file.py and run_file.py in the same working directory.
* Modify the correct data set work path in the mainfile.py file. The modification is as follows,
```Python
file_path = "smart_grid_dataset.csv" 
```

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
