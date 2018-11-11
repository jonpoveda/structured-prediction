## What to do ?

Fill template following the example ('plot_letters.py')

### Classifier
Classifier `y'`

Problem: estimate the coeficients `w`

```
y' = argmax{y} < w, phi(x,y) >  = argmax{y} g
```

where:
- x = observacions (8 images of chars of size 8x16)
- y = predictions / labels (a-z)
- w = coef
- phi() = jointfeatures

To solve `argmax`: BF, graph-cut, linear-prog, ...


### Graph
```
(y1) -- [p] -- (y2) -- [p] -- (yn)  unknowns (labels [1..26])
|                            |
[u]                          [u]  unary   p=pair-wise
|                            |
(o)                         (O)  observations
x1          x2  ...         xn  (image of 1st char, 2n char,...[16x8] matrix)    

(x^n ,y^n) = n-sample  n = [1...N], N samples
 
y^n = list of 8 numbers
x^n = collect of 8 binary images

u = size [128x26]
p = size [26x26]
`u` and `p` are shared along the graph (let's put it simple)
```

```
y' = argmax sum_{i=1..8} sum{p=1..26} sum{j=1..128} w_pj x_ij *1*_{yi=p} +
    + sum{i=1..l-1} sum{p=1..26} sum{q=1..26} wpq *1* {y_i=p, y_i+1 = q}

*1* = 1 if y_i = p
    = 0 else
```  

### How to estimate good coef?
```
w* = argmin{w1 Xi_1..Xi_n} 0.5 ||w||^2 + C/N sum Xi_n

y_n = groundtruth

such that:
    g(x_n, y_n;w) - g(x-n, y; w) >= Delta(y_n,y) - Xi_n
    g_for_rigth_solution - g_other_sol >= ... 

```
Xi are slack vars  (method: N-slack) ==> other methods: 1-slack, Frank-Wolfe .. (pystruct)

Use all samples (segments/ + more_samples/)

features = starting , ending points, length, ... (use some, which are the informatives?)
