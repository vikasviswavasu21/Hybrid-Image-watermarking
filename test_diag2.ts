import { Matrix } from "ml-matrix";

try {
    const d = Matrix.diag([1, 2]);
    console.log("diag works", d);
} catch (e) {
    console.error("diag failed", e);
}
