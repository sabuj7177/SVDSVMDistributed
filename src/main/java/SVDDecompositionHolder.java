import Jama.Matrix;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.SingularValueDecomposition;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;

import java.util.ArrayList;
import java.util.List;

public class SVDDecompositionHolder {
    Matrix U;
    Matrix S;
    Matrix mainMatrix;
//    JavaSparkContext sc;

    public SVDDecompositionHolder(Matrix mainMatrix) {
        this.mainMatrix = mainMatrix;
//        this.sc = sc;
    }

    /* Assuming ara has only one row */
    private double[] convert1dFrom2d(double[][] ara){
        int len = ara[0].length;
        double[] oneDAra = new double[len];
        for(int i=0;i<len;i++){
            oneDAra[i] = ara[0][i];
        }
        return oneDAra;
    }

    /* Assuming return ara will have only one row */
    private double[][] convert2dFrom1d(double[] ara){
        int len = ara.length;
        double[][] twoDAra = new double[1][len];
        for(int i=0;i<len;i++){
            twoDAra[0][i] = ara[i];
        }
        return twoDAra;
    }

    private List<Vector> convertMatrixToVectorList(Matrix m){
        int rowNumber = m.getRowDimension();
        int colNumber = m.getColumnDimension();
        List<Vector> data = new ArrayList<>();
        for(int i=0;i<rowNumber;i++){
            data.add(Vectors.dense(convert1dFrom2d(m.getMatrix(i,i,0,colNumber-1).getArray())));
        }
        return data;
    }

    private Matrix generateJamaMatrixFromRowMatrix(RowMatrix r){
        List<Vector> vs = r.rows().toJavaRDD().collect();
        int rowNum = vs.size();
        int colNum = vs.get(0).size();
        Matrix jamaMat = new Matrix(rowNum, colNum);
        for(int i=0;i<rowNum;i++) {
            jamaMat.setMatrix(i,i,0,colNum-1,new Matrix(convert2dFrom1d(vs.get(i).toArray())));
        }
        return jamaMat;
    }

    private Matrix generateDiagJamaMatrixFromVector(Vector v){
        return new Matrix(createDiagFrom1d(v.toArray()));
    }

    private double[][] createDiagFrom1d(double[] ara){
        int len = ara.length;
        double[][] diagAra = new double[len][len];
        for(int i=0;i<len;i++){
            diagAra[i][i] = ara[i];
        }
        return diagAra;
    }

    private static void PrintRowMatrix(RowMatrix r){
        List<Vector> vs = r.rows().toJavaRDD().collect();
        for(Vector v: vs) {
            System.out.println(v);
        }
    }

    public void calculateSVD(){
        List<Vector> data = convertMatrixToVectorList(mainMatrix);
        JavaRDD<Vector> rows = SingleSparkContext.sc.parallelize(data);
        RowMatrix mat = new RowMatrix(rows.rdd());
        SingularValueDecomposition<RowMatrix, org.apache.spark.mllib.linalg.Matrix> svd = mat.computeSVD(mainMatrix.getColumnDimension(), true, 1.0E-9d);
        RowMatrix rowU = svd.U();  // The U factor is a RowMatrix.
        Vector vecS= svd.s();     // The singular values are stored in a local dense vector.
//        org.apache.spark.mllib.linalg.Matrix V = svd.V();     // The V factor is a local dense matrix.

        U = generateJamaMatrixFromRowMatrix(rowU);
        S = generateDiagJamaMatrixFromVector(vecS);
    }

    public Matrix getU() {
        return U;
    }

    public Matrix getS() {
        return S;
    }
}
