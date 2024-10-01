Code for [Kabsch blog](https://www.atomhacker.xyz/2024/08/12/kabsch/).

<!-- more -->

## Kabsch Algorithm

Kabsch算法（又称Kabsch-Umeyama算法）是一种用于在两组对应点之间找到最佳刚体旋转的算法，目的是最小化两个点集之间的均方根误差（RMSD, Root Mean Square Deviation），该算法在分子模拟、图机器学习（or GNN）等领域中非常有用。

### 算法原理

Kabsch算法的目标是给定两个质心相同的点集，找到一个旋转矩阵 $R$ 来使得其中一个点集旋转后与另外一个点集之间的欧式距离最小。算法的核心思想是通过对点集进行去质心化和对协方差矩阵进行奇异值分解（SVD）来找到最佳的刚体旋转矩阵，从而最小化点集之间的距离。

**注意前提是点群的点已经是一一对应的**

假设有两个任意点集$P$和$Q$，$P \in \mathbb{R}^{N \times D}$，$Q$的维度与$P$一致，其中$N$为节点的个数，$D$为节点特征的维度，例如分子的笛卡尔坐标的点集为$\mathbb{R}^{N \times 3}$。则Kabsch算法的目标为找到旋转矩阵$R$和平移向量$t$使得下式最小：

$$
\min\sum_{i=1}^{N} \left\| {q}_i - R (p_i + t) \right\|^2 
$$

除基本对齐的要求外，有需要也可以为点集中的点加入权重$\omega$，这在许多问题中有重要作用。例如需要对大分子进行align，我们可以添加原子的相对原子质量作为权重，一定程度忽略H原子对结构对齐的影响，因为H原子很多情况下可以自由旋转，则优化目标可变为下式：
$$
\min\sum_{i=1}^{N} \omega_i \left\| q_i - R (p_i + t) \right\|^2 
$$

- 去质心化（Centering the Point Sets）

首先计算点集$P$和$R$的质心：
$$
c_P = \frac{1}{N} \sum_{i=1}^{N} p_i
$$

$$
c_Q = \frac{1}{N} \sum_{i=1}^{N} q_i
$$

通过减去质心坐标，将两个点群平移到相同的质心。

$$
P' = P - c_P
$$

$$
Q' = Q - c_Q
$$

- 计算协方差矩阵（Covariance Matrix）幷进行奇异值分解（SVD）

通过两个去质心化点群的外积求得点群之间的协方差矩阵$H$：

$$
H = P'^TQ'=\sum_{i=1}^{n} {p'_i}^T q'_i \in \mathbb{R}^{D \times D}
$$

然后对$H$进行奇异值分解：

$$
H = U \Sigma V^T
$$

再由下式即可求得旋转矩阵$R$：

$$
R = VU^T
$$

此处要注意若行列式的值$det(VU^T)$为负数，则说明出现了$R$变换出现了反射（Reflection），因此需要对$V$最后一个奇异值进行修正，令$B=diag(1,\ 1,\ sign(det(VU^T)))$，修正后的旋转矩阵$R$如下：

$$
R = VBU^T
$$

例如当$det(VU^T) < 0$，则有

$$
B = \begin{pmatrix} 
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & \text{sign}(det(VU^T))
\end{pmatrix} = \begin{pmatrix} 
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & -1
\end{pmatrix}
$$

$$
R = V \begin{pmatrix} 
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & -1
\end{pmatrix} U^T
$$

- 进行Align

计算平移向量$t$,即求两个质心之间的距离向量：

$$
t = c_Q - c_P
$$

$$
P_{aligned} = P' R
$$

然后可简单地求出RMSD：

$$
RMSD = \sqrt{\frac{1}{N}\sum_{i=1}^N\left\| q'_i - p_{aligned, i} \right\| ^ 2}
$$

## 代码实现

代码基于Python的PyTorch框架实现，同时也提供了NumPy和JAX框架，以及Rust的实现代码。

### PyTorch实现

```python
import torch

def kabsch_torch(P, Q):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD.
    :param P: A Nx3 matrix of points
    :param Q: A Nx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Compute centroids
    centroid_P = torch.mean(P, dim=0)
    centroid_Q = torch.mean(Q, dim=0)

    # Optimal translation
    t = centroid_Q - centroid_P

    # Center the points
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    H = torch.matmul(p.transpose(0, 1), q)

    # SVD
    U, S, Vt = torch.linalg.svd(H)

    # Validate right-handed coordinate system
    if torch.det(torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1))) < 0.0:
        Vt[:, -1] *= -1.0

    # Optimal rotation
    R = torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1))

    # RMSD
    rmsd = torch.sqrt(torch.sum(torch.square(torch.matmul(p, R.transpose(0, 1)) - q)) / P.shape[0])

    return R, t, rmsd
```

### NumPy实现

```python
import numpy as np

def kabsch_numpy(P, Q):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD.

    :param P: A Nx3 matrix of points
    :param Q: A Nx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Compute centroids
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    # Optimal translation
    t = centroid_Q - centroid_P

    # Center the points
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    H = np.dot(p.T, q)

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Validate right-handed coordinate system
    if np.linalg.det(np.dot(Vt.T, U.T)) < 0.0:
        Vt[-1, :] *= -1.0

    # Optimal rotation
    R = np.dot(Vt.T, U.T)

    # RMSD
    rmsd = np.sqrt(np.sum(np.square(np.dot(p, R.T) - q)) / P.shape[0])

    return R, t, rmsd
```

### JAX实现

```python
import jax.numpy as jnp


def kabsch_jax(P, Q):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD.

    :param P: A Nx3 matrix of points
    :param Q: A Nx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Compute centroids
    centroid_P = jnp.mean(P, axis=0)
    centroid_Q = jnp.mean(Q, axis=0)

    # Optimal translation
    t = centroid_Q - centroid_P

    # Center the points
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    H = jnp.dot(p.T, q)

    # SVD
    U, S, Vt = jnp.linalg.svd(H)

    # Validate right-handed coordinate system
    if jnp.linalg.det(jnp.dot(Vt.T, U.T)) < 0.0:
        Vt[-1, :] *= -1.0

    # Optimal rotation
    R = jnp.dot(Vt.T, U.T)

    # RMSD
    rmsd = jnp.sqrt(jnp.sum(jnp.square(jnp.dot(p, R.T) - q)) / P.shape[0])

    return R, t, rmsd
```
### Rust实现

Cargo.toml中添加依赖项nalgebra:

```toml
[dependencies]
nalgebra = "0.31" 
```

```rust
use nalgebra::{Matrix3, Vector3, DMatrix, SVD};

/// Computes the optimal rotation and translation to align two sets of points (P -> Q),
/// and their RMSD.
///
/// # Arguments
///
/// * `P` - A Nx3 matrix of points
/// * `Q` - A Nx3 matrix of points
///
/// # Returns
///
/// A tuple containing the optimal rotation matrix (Matrix3), 
/// the optimal translation vector (Vector3), and the RMSD (f64).
fn kabsch_rust(P: &DMatrix<f64>, Q: &DMatrix<f64>) -> (Matrix3<f64>, Vector3<f64>, f64) {
    assert_eq!(P.ncols(), 3);
    assert_eq!(Q.ncols(), 3);
    assert_eq!(P.nrows(), Q.nrows());

    let n = P.nrows();

    // Compute centroids
    let centroid_P = P.column_mean();
    let centroid_Q = Q.column_mean();

    // Optimal translation
    let t = centroid_Q - centroid_P;

    // Center the points
    let p = P - centroid_P;
    let q = Q - centroid_Q;

    // Compute the covariance matrix
    let h = p.transpose() * q;

    // Perform SVD
    let svd = SVD::new(h, true, true);
    let u = svd.u.unwrap();
    let v_t = svd.v_t.unwrap();

    // Validate right-handed coordinate system
    let mut d = v_t.transpose() * u.transpose();
    if d.determinant() < 0.0 {
        d[(2, 2)] *= -1.0;
    }

    // Optimal rotation
    let r = v_t.transpose() * d * u.transpose();

    // Compute RMSD
    let transformed_p = p * r.transpose();
    let rmsd = ((transformed_p - q).norm_squared() / n as f64).sqrt();

    (r, t, rmsd)
}

fn main() {
    // Example usage
    let p = DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    let q = DMatrix::from_row_slice(3, 3, &[0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);

    let (rotation, translation, rmsd) = kabsch_rust(&p, &q);

    println!("Rotation matrix:\n{}", rotation);
    println!("Translation vector:\n{}", translation);
    println!("RMSD: {}", rmsd);
}
```

## 测试

以乙醇为例，创建两个甲醇分子对象，进行对齐并计算$RMSD$。

- 定义函数生成分子构象

```python
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def generate_methanol(seed: int):
    # 1. 根据SMILES创建分子
    smiles = 'CO'  # 甲醇的SMILES
    mol = Chem.MolFromSmiles(smiles)
    # 2. 添加H原子
    mol = Chem.AddHs(mol)
    # 3. 生成三维坐标
    mol1 = AllChem.EmbedMolecule(mol, randomSeed=seed)
    # 4. 优化几何结构
    AllChem.UFFOptimizeMolecule(mol)
    
    return mol

def get_coords(mol: Chem.Mol):
    coords = []
    conf = mol.GetConformer()
    print(f"Number of atoms: {mol.GetNumAtoms()}")
    # 打印坐标信息
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        print(f"Atom {atom.GetSymbol()} {i}: {pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f}")
        coords.append(np.array([pos.x, pos.y, pos.z]))

    return np.array(coords)
```

- 生成乙醇原子的笛卡尔坐标作为点群$P$和$Q$

```python
mol1 = generate_methanol(42)
P = get_coords(mol1)
mol2 = generate_methanol(47)
Q = get_coords(mol2)
```

```python
Number of atoms: 6
Atom C 0: -0.36, 0.01, -0.02
Atom O 1: 0.91, -0.53, -0.26
Atom H 2: -0.55, 0.07, 1.07
Atom H 3: -0.43, 1.02, -0.48
Atom H 4: -1.13, -0.65, -0.48
Atom H 5: 1.56, 0.08, 0.17
Number of atoms: 6
Atom C 0: -0.36, 0.00, -0.02
Atom O 1: 0.92, -0.53, -0.24
Atom H 2: -0.55, 0.11, 1.07
Atom H 3: -0.44, 0.99, -0.52
Atom H 4: -1.12, -0.68, -0.46
Atom H 5: 1.55, 0.11, 0.17
```

故意将随机种子固定是为了保证当前的原子次序是一一对应的。

- 随机对$Q$进行旋转和平移

```python
def random_rotation_matrix(dim=3):
    # 随机生成一个方阵
    random_matrix = np.random.randn(dim, dim)
    # 使用QR分解生成一个正交矩阵
    q, r = np.linalg.qr(random_matrix)
    # 为确保行列式为 1，调整符号
    d = np.linalg.det(q)
    q = q * np.sign(d)
    
    return q

R_random = random_rotation_matrix()
t = np.random.randn(3)
Q = Q @ R_random + t

print('P:', P)
print('Q:', Q)
```

```python
P: [[-0.35770023  0.00759022 -0.02148174]
 [ 0.90873557 -0.53499245 -0.26111898]
 [-0.54683347  0.07179144  1.07210873]
 [-0.43376811  1.01934375 -0.47579473]
 [-1.12699742 -0.64793055 -0.47895646]
 [ 1.55656366  0.08419759  0.16524319]]
Q: [[-0.86317234 -0.72913679 -1.05250391]
 [-1.59568206  0.46172438 -1.02150308]
 [-1.24596703 -1.39410603 -1.85693876]
 [ 0.21356653 -0.51279279 -1.22472573]
 [-0.9715575  -1.2485602  -0.07820333]
 [-1.46667556  0.88522581 -1.9096848 ]]
```

- 不对齐直接求$RMSD$

```python
def rmsd(P, Q):
    return np.sqrt(np.mean(np.sum((P - Q) ** 2, axis=1)))
print(rmsd(P, Q))
```

```python
2.5456441356883777
```

- 使用Kabsch算法进行对齐并求得$RMSD$

```python
R_, t_, rmsd = kabsch_numpy(P, Q)
print(rmsd)
```

```python
1.881049755021318e-06
```

注意此处求得的旋转矩阵和开始QR分解随机生成的正交矩阵（$R$）并不是相同或互逆的，平移向量也不相同或者相反，这是因为最开始对点群$Q$旋转时质点并不在原点，$Q$做正交矩阵的变换相当于相对原点旋转，而求出的旋转矩阵是质心在原点时的自旋对应的变换。

## Scource Code

[Github: An implementation of Kabsch algorithm.](https://github.com/zEuler1027/Kabsch)

## 参考文章

> 1. Kabsch W. A discussion of the solution for the best rotation to relate two sets of vectors[J]. Acta Crystallographica Section A: Crystal Physics, Diffraction, Theoretical and General Crystallography, 1978, 34(5): 827-828.
