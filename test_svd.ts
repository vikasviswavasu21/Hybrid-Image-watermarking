import { Matrix, SingularValueDecomposition } from "ml-matrix";
const m = new Matrix([[1, 2], [3, 4]]);
const svd = new SingularValueDecomposition(m);
console.log(Object.keys(svd));
console.log(svd.leftSingularVectors);
console.log(svd.diagonal);
console.log(svd.rightSingularVectors);
