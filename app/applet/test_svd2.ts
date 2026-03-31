import { Matrix, SingularValueDecomposition } from "ml-matrix";
const m = new Matrix([[1, 2], [3, 4]]);
const svd = new SingularValueDecomposition(m);
const U = svd.leftSingularVectors;
const S = Matrix.diag(svd.diagonal);
const V = svd.rightSingularVectors;
const reconstructed = U.mmul(S).mmul(V.transpose());
console.log(reconstructed);
