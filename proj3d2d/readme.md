# Assigment 1 Report

+ Language : python 
+ third-party package : numpy 
+ execution : command ```python project3d2d.py```
+ The result is at folder ```projection```

## Method :
### Projection matrix:

3D pair : $\begin{bmatrix}\cdots & \cdots & \cdots \\ X_i & Y_i & Z_i \\ \cdots & \cdots & \cdots \end{bmatrix}_{N\times 3}$ ,2D pair : $\begin{bmatrix} \cdots & \cdots \\ x_i & y_i \\ \cdots & \cdots \end{bmatrix}_{N\times 2}$ 

$A_{2N\times 12} = \begin{bmatrix}
        \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots  \\
        X_i & Y_i & Z_i & 1 & 0 & 0 & 0 & 0 &-x_iX_i & -x_iY_i & -x_iZ_i & -x_i \\ 
        0 & 0 & 0 & 0 & X_i & Y_i & Z_i & 1 &-y_iX_i & -y_iY_i & -y_iZ_i & -y_i \\ 
        \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots 
    \end{bmatrix}$

Approximate $P$ by optimizing the problem :\
   $\underset{P}{\textbf{argmin}}||(A^TA)P-0||_2,\text{s.t }P^TP=1$ 
- Letting $P$ be the eigen vector corresponding to minimal eigen value $\lambda$, we minmize the objective function to $\lambda$.
    
By using ```eigenvalues, eigenvectors = np.linalg.eig(A.T@A)``` to find eigen value and eigen vector for $A^TA$.

Find minmal eigen value 's position using ```np.argmin(eigenvalues)``` and selection that column from ```eigenvectors``` to get $P$.

Finally, reshape $P$ to $3 \times 4$ to get projection matrix __P__.

###  calibration matrix (C), Rotation matrix (Rotation) and translation matrix (T)
$$P=C \times \left[\begin{array}{c|c}
    \text{Rotation} & T
\end{array}\right]=\left[\begin{array}{c|c}
C\times \text{Rotation} & C \times T \end{array}\right] =
\left[\begin{array}{c|c}
    P_{33} & P{_4}
\end{array}\right]$$

$A = QR=\text{orthonormal}\times \text{Upper}$. However, $P_{33}=\text{Upper}\times\text{orthonormal}$. 

Thus, we need to invert the $P_{33}$ then can apply $QR$-decomposition :

$A = QR \Rightarrow A^{-1}=R^{-1}Q^{-1}=R^{-1}Q^{T}$

$P_{33}^{-1} = QR \Rightarrow P_{33}=R^{-1}Q^{T}=C\times\text{Rotation}$
1. $C = R^{-1}$ 
2. $\text{Rotation} = Q^T$.
3. $T=C^{-1}P_4$.

Using ```np.linalg.qr()``` to decompose the inverse of upper 3x3 part of __P__ ($QR = P_{33}^{-1}$) and appling equations 1. 2. to get $C$ and $\text{Rotation}$. After that, get $T$ using equation 3


### Average projection Error:
$\text{project}_{h,N \times 3} = \begin{bmatrix}
    \cdots & \cdots & \cdots \\ 
    x_h & y_h & z_h \\ 
    \cdots & \cdots & \cdots \\
\end{bmatrix} = P_{3 \times 4}\times X_{N\times (3+1)}^T$

Translate back from homogeneous coordinate system to cartesian coordinate system, divided by the last element for each sample :
$\begin{bmatrix}
\cdots & \cdots \\ 
\frac{x_h}{z_h} & \frac{y_h}{z_h} \\ 
\cdots & \cdots \\ 
\end{bmatrix}$

Then calculate MSE between groundtruth $\begin{bmatrix}
\cdots & \cdots \\ 
\frac{x_h}{z_h} & \frac{y_h}{z_h} \\ 
\cdots & \cdots \\ 
\end{bmatrix} , \begin{bmatrix}
    \cdots & \cdots \\ 
    \hat{x} & \hat{y} \\ 
    \cdots & \cdots \\
\end{bmatrix}$

Average projection error for sample data :  __0.427089376559275__
