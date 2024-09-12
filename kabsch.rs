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
