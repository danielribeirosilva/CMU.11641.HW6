
1. This dataset (and the associated evaluation tools) was released by MSR Asia . Please abide by the agreement policies described in the file LETOR-agreement.txt for using the dataset.

2. The format of the dataset, the features used are explained in the given PDF file.

3. The Evaluation Program prints PREC, MAP and NDCG (the latter and at each position from 1 to 16)

Optional:
For faster convergence, you might you want to L2 normalize each query-doc pair  i.e. divide all the features of query-doc vector by the L2 norm of the vector. For example, if the vector is 

0 qid:34 1:3.4 2:11 3:0.4

The L2 normalized vector is the vector whose elements are divided by sqrt(3.4*3.4+11*11+.4*.4) = 11.52

0 qid:34 1:.2951 2:.9548 3:.0347


