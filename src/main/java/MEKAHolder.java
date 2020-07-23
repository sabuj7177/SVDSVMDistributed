import Jama.Matrix;
import Jama.SingularValueDecomposition;
import breeze.linalg.svd;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import jep.JepException;
import jep.MainInterpreter;
import jep.NDArray;
import jep.SharedInterpreter;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.storage.StorageLevel;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static java.lang.Math.min;
import static java.lang.Math.pow;

class MEKAHolder implements Serializable {
    private Matrix trainData;
    private Matrix trainLabel;
    private Matrix testData;
    private Matrix testLabel;
    private Matrix trainDataWithLabel;
    private String inputDataPath;
    private String inputLabelPath;
    private int numOfClusters;
    private int numOfCols;
    private int numOfTotalData;
    private int kMeansIterations;
    private int targetRank;
    private double gamma;
    private double eta;
    private double splitPartition;
    int flag = 0;

    MEKAHolder(String inputDataPath,
               String inputLabelPath,
               int numOfClusters,
               int numOfTotalData,
               int numOfCols,
               int kMeansIterations,
               int targetRank,
               double gamma,
               double eta,
               double splitPartition)
    {
        this.inputDataPath = inputDataPath;
        this.inputLabelPath = inputLabelPath;
        this.numOfClusters = numOfClusters;
        this.numOfCols = numOfCols;
        this.numOfTotalData = numOfTotalData;
        this.kMeansIterations = kMeansIterations;
        this.targetRank = targetRank;
        this.gamma = gamma;
        this.eta = eta;
        this.splitPartition = splitPartition;
    }

    Matrix getTrainData() {
        return trainData;
    }

    Matrix getTrainLabel() {
        return trainLabel;
    }

    Matrix getTestData() {
        return testData;
    }

    Matrix getTestLabel() {
        return testLabel;
    }

    Matrix getTrainDataWithLabel() {
        return trainDataWithLabel;
    }

    private void print(Matrix m, int startRow, int endRow, int startColumn, int endColumn) {
        for (int i = startRow; i < endRow; i++) {
            for (int j = startColumn; j < endColumn; j++) {
                System.out.print(m.get(i, j) + " ");
            }
            System.out.println();
        }
    }

    private void print(Matrix m) {
        for (int i = 0; i < m.getRowDimension(); i++) {
            for (int j = 0; j < m.getColumnDimension(); j++) {
                System.out.print(m.get(i, j) + " ");
            }
            System.out.println();
        }
    }

    private JavaRDD<Matrix> getData(JavaRDD<String> stringDataRDD){
        return stringDataRDD.map((Function<String, Matrix>) s -> {
            String[] splitArray = s.trim().split(" ");
            if(splitArray.length != numOfCols){
                throw new CustomException("Input array dimension should be equal to number of columns");
            }
            double[][] rowArray = new double[1][numOfCols];
            for (int i = 0; i < numOfCols; i++) {
                rowArray[0][i] = Double.parseDouble(splitArray[i]);
            }
            return new Matrix(rowArray);
        }
        ).persist(StorageLevel.MEMORY_ONLY_SER());
    }

    private Matrix getDataMatrixFromRDD(JavaRDD<Matrix> dataRDD){
        Matrix dataMatrix = new Matrix(numOfTotalData, numOfCols);
        int rowIterator = 0;
        for (Matrix line : dataRDD.take(numOfTotalData)) {
            dataMatrix.setMatrix(rowIterator, rowIterator, 0, numOfCols - 1, line);
            rowIterator++;
        }
        return dataMatrix;
    }

    private JavaRDD<Double> getLabel(JavaRDD<String> stringLabelRDD){
        return stringLabelRDD.map((Function<String, Double>) s -> {
            String[] splitArray = s.trim().split(" ");
            return Double.parseDouble(splitArray[0]);
        }
        ).persist(StorageLevel.MEMORY_ONLY_SER());
    }

    private Matrix getLabelMatrixFromRDD(JavaRDD<Double> labelRDD){
        Matrix labelMatrix = new Matrix(numOfTotalData,1);
        int rowIterator = 0;
        for (double line : labelRDD.take(numOfTotalData)) {
            labelMatrix.set(rowIterator, 0, line);
            rowIterator++;
        }
        return labelMatrix;
    }

    private Matrix[] getKMeansClusterCenters(JavaRDD<String> stringDataRDD){
        JavaRDD<Vector> parsedData = stringDataRDD.map((Function<String, Vector>) s -> {
            String[] splitArray = s.split(" ");
            double[] values = new double[splitArray.length];
            for (int i = 0; i < splitArray.length; i++) {
                values[i] = Double.parseDouble(splitArray[i]);
            }
            return Vectors.dense(values);
        }).cache();

        KMeansModel clusters = KMeans.train(parsedData.rdd(), numOfClusters, kMeansIterations);
        Vector[] centers = clusters.clusterCenters();
        Matrix[] clusterCenters = new Matrix[numOfClusters];
        for(int i=0;i<numOfClusters;i++){
            clusterCenters[i] = new Matrix(centers[i].toArray(), numOfCols);
        }
        return clusterCenters;
    }

    private Matrix getFinalDataDistanceFromClusters(Matrix[] clusterCenters, Matrix dataMatrix){
        Matrix finalDistance = new Matrix(numOfTotalData, numOfClusters);
        for(int i=0;i<numOfTotalData;i++){
            Matrix rowMatrix = dataMatrix.getMatrix(i,i,0, numOfCols-1);
            for(int j=0;j<numOfClusters;j++){
                finalDistance.set(i,j, pow((rowMatrix.transpose()).minus(clusterCenters[j]).normF(),2));
            }
        }
        return finalDistance;
    }

    private int getMinIndexOfRowMatrix(Matrix m){
        double minDis = m.get(0,0);
        int minIndex = 0;
        for(int i=1;i<m.getColumnDimension();i++){
            if(m.get(0,i)< minDis){
                minIndex = i;
                minDis = m.get(0,i);
            }
        }
        return minIndex;
    }

    private Matrix getFinalDataIndex(Matrix finalDistance){
        Matrix finalIndex = new Matrix(numOfTotalData, 1);
        for(int i=0;i<numOfTotalData;i++){
            finalIndex.set(i,0, getMinIndexOfRowMatrix(finalDistance.getMatrix(i,i,0,numOfClusters-1)));
        }
        return finalIndex;
    }

    private double eucledianDistance1d(Matrix x, Matrix y){
        return x.minus(y).normF();
    }

    private double squaredEucledianDistance1d(Matrix x, Matrix y){
        return Math.pow(eucledianDistance1d(x,y),2);
    }

    private Matrix squaredEucledianDistance2d(Matrix x, Matrix y){
        int xLen = x.getRowDimension();
        int xCol = x.getColumnDimension();
        int yLen = y.getRowDimension();
        int yCol = y.getColumnDimension();
        Matrix distanceMatrix = new Matrix(xLen, yLen);
        for(int i=0;i<xLen;i++){
            for(int j=0;j<yLen;j++){
                distanceMatrix.set(i,j,
                        squaredEucledianDistance1d(x.getMatrix(i,i,0,xCol-1).transpose(),
                                y.getMatrix(j,j,0,yCol-1).transpose()));
            }
        }
        return distanceMatrix;
    }

    private Matrix elementWiseExp(Matrix x){
        int xLen = x.getRowDimension();
        int xCol = x.getColumnDimension();
        Matrix expMatrix = new Matrix(xLen, xCol);
        for(int i=0;i<xLen;i++){
            for(int j=0;j<xCol;j++){
                expMatrix.set(i,j, Math.exp(x.get(i,j)));
            }
        }
        return expMatrix;
    }

    private Matrix squareDistanceBetweenMatrix(Matrix x, Matrix y, double gamma){
        Matrix distanceMatrix = squaredEucledianDistance2d(x,y);
        distanceMatrix = distanceMatrix.times(-1.0 * gamma);
        return elementWiseExp(distanceMatrix);
    }

    private Matrix getSubMatrix(Matrix m, List<Integer> elements){
        int colDimension = m.getColumnDimension();
        Matrix subMatrix = new Matrix(elements.size(), colDimension);
        for(int i=0;i<elements.size();i++){
            subMatrix.setMatrix(i,i,0,colDimension-1, m.getMatrix(elements.get(i), elements.get(i), 0, colDimension-1));
        }
        return subMatrix;
    }

    private List<Integer> getRandomPermutation(int n){
        List<Integer> shuffleList = new ArrayList<>();
        for(int i=0;i<n;i++){
            shuffleList.add(i);
        }
        java.util.Collections.shuffle(shuffleList);
        return shuffleList;
    }

    private static Matrix getElementWisePower(Matrix m, double p){
        for(int i=0;i<m.getRowDimension();i++){
            for(int j=0;j<m.getColumnDimension();j++){
                if(m.get(i,j)!=0.0) {
                    m.set(i, j, pow(m.get(i, j), p));
                }
            }
        }
        return m;
    }

    private SVDHolder getPythonSVD(Matrix m) throws CustomException, JepException {
        SVDHolder svdHolder = new SVDHolder();
        int row = m.getRowDimension();
        int col = m.getColumnDimension();
        int arrayLength = row * col;
        double[] f = new double[arrayLength];
        for(int i=0; i<row; i++){
            for (int j=0; j<col; j++){
                f[i*col+j] = m.get(i,j);
            }
        }
        try(SharedInterpreter interp = new SharedInterpreter()) {
            NDArray<double[]> nd = new NDArray<>(f, row, col);
            interp.eval("from scipy.linalg import svd");
            interp.set("x", nd);
            interp.eval("u, s, vh = svd(x)");
            Object result1 = interp.getValue("u");
            if (result1 instanceof NDArray){
                int[] dimensions = ((NDArray)result1).getDimensions();
                int returnMatrixRow = dimensions[0];
                int returnMatrixCol = dimensions[1];
                double[] returnMatrix = (double[])(((NDArray)result1).getData());
                Matrix U = new Matrix(returnMatrixRow, returnMatrixCol);
                for(int i=0; i<returnMatrixRow; i++){
                    for(int j=0; j<returnMatrixCol; j++){
                        U.set(i,j,returnMatrix[i*returnMatrixCol+j]);
                    }
                }
                svdHolder.setU(U);
            }
            else{
                throw new CustomException("U is not available");
            }

            Object result2 = interp.getValue("s");
            if (result2 instanceof NDArray){
                int[] dimensions = ((NDArray)result2).getDimensions();
                int returnMatrixRow = dimensions[0];
//                int returnMatrixCol = dimensions[1];
                double[] returnMatrix = (double[])(((NDArray)result2).getData());
                Matrix S = new Matrix(returnMatrixRow, returnMatrixRow);
                for(int i=0; i<returnMatrixRow; i++){
                    S.set(i,i,returnMatrix[i]);
                }
                svdHolder.setS(S);
            }
            else{
                throw new CustomException("S is not available");
            }

        } catch (JepException e) {
            e.printStackTrace();
        }
        return svdHolder;
    }

    private Matrix NysTrain(Matrix dataPartitionMatrix, int m, int k, double gamma) throws CustomException, JepException {
        int n = dataPartitionMatrix.getRowDimension();
        List<Integer> shuffleList = getRandomPermutation(n);
        List<Integer> subList = shuffleList.subList(0,m);
        Matrix subMatrix = getSubMatrix(dataPartitionMatrix, subList);
        Matrix W = squareDistanceBetweenMatrix(subMatrix, subMatrix, gamma);
        Matrix E = squareDistanceBetweenMatrix(dataPartitionMatrix, subMatrix, gamma);
//        DoubleMatrix2D pColt_matrix = new DenseDoubleMatrix2D(W.getRowDimension(),W.getColumnDimension());
//        pColt_matrix.assign(W.getArray());
//        cern.colt.matrix.linalg.SingularValueDecomposition colt_svd = new cern.colt.matrix.linalg.SingularValueDecomposition(pColt_matrix);
//        SVDDecompositionHolder svd = new SVDDecompositionHolder(W);
//        svd.calculateSVD();
//        SingularValueDecomposition svd = new SingularValueDecomposition(W);
        SVDHolder svdHolder = getPythonSVD(W);
        Matrix U = svdHolder.getU();
        Matrix S = svdHolder.getS();
        Matrix invSqrtS = getElementWisePower(S, -0.5);
        Matrix croppedInvSqrtS = invSqrtS.getMatrix(0, k-1, 0, k-1);
        Matrix croppedU = U.getMatrix(0, U.getRowDimension()-1, 0, k-1);
        return E.times(croppedU.times(croppedInvSqrtS));
    }

    private List<Integer> getElementsByListIndex(List<Integer> elementsList, List<Integer> indexList){
        List<Integer> outputList = new ArrayList<>();
        for (Integer index : indexList) {
            outputList.add(elementsList.get(index));
        }
        return outputList;
    }

    private Matrix elementWiseMultiplication(Matrix A, Matrix B) throws CustomException {
        int aColNum = A.getColumnDimension();
        if(A.getRowDimension() != B.getRowDimension()){
            throw new CustomException("A and B data size is not same");
        }
        if(B.getColumnDimension() != 1){
            throw new CustomException("B must have one column");
        }
        Matrix C = new Matrix(A.getRowDimension(),A.getColumnDimension());
        for(int i=0;i<A.getRowDimension();i++){
            C.setMatrix(i,i,0,aColNum-1,A.getMatrix(i,i,0,aColNum-1).times(B.get(i,0)));
        }
        return C;
    }

    private Matrix[] getTemporaryMatrix(){
        double[][] tempArray1 = {{ 3.23574494e-01,  2.00183936e-01,  1.77191907e-01,  1.54583078e-01,
        1.44466585e-01,  7.04552422e-01,  5.11189454e-02,  1.80104231e-02,
        2.88933170e-02,  7.25781729e-02,  4.81299816e-02,  2.29920294e-04,
        4.59840589e-04, 2.02942980e-01, 1.92213366e-01, 1.92903127e-01,
        2.06085224e-01, 2.05855303e-01, 2.21566524e-01, 3.67795831e-01,
        6.58338443e-02, 7.31636973e-14, 1.45616186e-02, 5.24218271e-02,
        6.01624770e-02, 2.34518700e-02, 2.38350705e-02, 2.48313918e-02,
        6.36112814e-02, 7.12752912e-03, 4.72869405e-02, 1.11128142e-02,
        1.37952177e-02, 2.60576334e-03, 2.08767627e-01, 7.31636973e-14,
        3.67795831e-01, 1.12584304e-01, 3.10852238e-01, 8.66033109e-02,
        2.16738197e-01, 5.82004905e-01, 4.74402207e-02, 4.41446965e-02,
        2.26088289e-02, 4.59840589e-04, 3.70938075e-02, 6.33047210e-02,
        1.31054568e-01, 1.14040466e-01, 1.12507664e-01, 1.66922134e-01,
        3.89331698e-02, 4.19987738e-02, 1.52973636e-01, 1.99264255e-02,
        2.31453096e-02, 7.51072961e-03, 1.38718578e-02, 2.29920294e-04,
        8.14684243e-02, 2.71459227e-01, 1.16573418e-15, 4.42903127e-01,
        4.10024525e-02, 1.63166769e-01, 8.25107296e-01, 3.60974862e-02,
        1.07296137e-02, 1.02697731e-02, 1.17795831e-01, 5.65297364e-01,
        4.34702636e-01, 9.39071122e-01, 6.09288780e-02, 9.63979154e-01,
        3.60208461e-02, 2.57204169e-01, 8.53770693e-02, 4.31713673e-01,
        6.87461680e-02, 1.56958921e-01, 8.84426732e-01, 4.59840589e-04,
        2.83568363e-03, 3.83200490e-03, 3.67872471e-03, 5.05824647e-03,
        7.66400981e-04, 2.68240343e-03, 1.76272226e-03, 5.36480687e-04,
        2.60576334e-03, 1.99264255e-03, 3.14224402e-03, 1.53280196e-03,
        6.89760883e-04, 7.35744942e-03, 1.91600245e-03, 1.68608216e-03,
        2.91232373e-03, 2.68240343e-03, 2.18424280e-02, 9.96321275e-04,
        6.89760883e-04, 1.07296137e-03, 3.14224402e-03, 6.89760883e-04,
        7.66400981e-04, 2.06928265e-03, 1.99264255e-03, 2.06928265e-03,
        4.59840589e-04, 2.98896383e-03, 1.22624157e-03, 3.83200490e-04,
        7.66400981e-04, 3.06560392e-04, 4.59840589e-03, 6.89760883e-04,
        1.07296137e-03, 6.13120785e-04, 7.66400981e-05}};
        double[][] tempArray2 = {{ 3.45095428e-02, 1.44141145e-01, 2.44562805e-01, 2.69862406e-01,
        3.06924101e-01, 6.35042166e-01, 1.11850866e-01, 6.67998225e-02,
        3.55082113e-02, 7.13493120e-02, 4.46071904e-02, 2.21926320e-04,
                        -2.22803546e-17, 2.04727031e-01, 2.07833999e-01, 1.94851309e-01,
        1.92964936e-01, 1.99622725e-01, 2.73413227e-01, 2.76520195e-01,
        3.50643586e-02, 2.66453526e-14, 4.28317798e-02, 4.24988904e-02,
        6.62450067e-02, 2.30803373e-02, 3.71726587e-02, 1.20949845e-02,
        9.90901021e-02, 8.32223702e-03, 3.50643586e-02, 2.97381269e-02,
        1.69773635e-02, 1.88637372e-03, 1.69662672e-01, 2.66453526e-14,
        2.76520195e-01, 1.08743897e-01, 4.45073236e-01, 9.98779405e-01,
        2.21926320e-04, 3.55271368e-14, 2.36616282e-15, 3.32889481e-04,
        2.21926320e-04, 4.43852641e-04, 3.16245007e-02, 1.49911229e-01,
        3.37328007e-02, 1.21282734e-01, 1.98735020e-01, 1.91855304e-01,
        2.78517532e-02, 4.78251221e-02, 4.53839325e-02, 3.66178429e-02,
        5.21526853e-02, 1.10963160e-04, 2.80736795e-02, 2.21926320e-04,
        7.76742122e-04, 2.44118953e-03, 9.89125610e-01, 9.98668442e-04,
        6.10297381e-03, 5.54815801e-04, 9.03905903e-01, 3.79494008e-02,
        5.77008433e-03, 6.54682645e-03, 4.58277852e-02, 5.54815801e-04,
        9.99445184e-01, 8.63071460e-01, 1.36928540e-01, 9.28317798e-01,
        7.16822015e-02, 7.46782068e-02, 3.79494008e-02, 4.50510431e-01,
        1.02197071e-01, 3.34664891e-01, 8.79715934e-01, 6.65778961e-04,
        2.77407901e-03, 2.77407901e-03, 4.66045273e-03, 4.66045273e-03,
        1.10963160e-04, 6.54682645e-03, 2.44118953e-03, 1.44252108e-03,
        2.55215269e-03, 3.43985797e-03, 3.32889481e-03, 2.10830004e-03,
        1.40404181e-17, 7.76742122e-03, 3.43985797e-03, 1.99733688e-03,
        1.88637372e-03, 1.77541056e-03, 2.54105637e-02, 1.55348424e-03,
        3.32889481e-04, 9.98668442e-04, 1.22059476e-03, 4.43852641e-04,
        8.87705282e-04, 2.33022636e-03, 5.54815801e-04, 1.33155792e-03,
        2.21926320e-04, 1.10963160e-03, 9.98668442e-04, 2.21926320e-04,
        2.21926320e-04, 9.98668442e-04, 2.77407901e-03, 4.43852641e-04,
        4.43852641e-04, 8.87705282e-04, -5.66495635e-18}};
        double[][] tempArray3 = {{ 2.71684704e-01, 1.97221774e-01, 2.00775319e-01, 1.55871426e-01,
        1.74446778e-01, 7.81941528e-01, 5.08803101e-02, 1.45372315e-02,
        2.45517687e-02, 4.52269423e-02, 2.48748183e-02, 9.69148764e-04,
        1.61524794e-04, 1.85591988e-01, 2.02713617e-01, 2.07720885e-01,
        2.02067517e-01, 2.01905993e-01, 2.35089725e-14, 2.35367281e-14,
                -4.50334214e-15, 1.00000000e+00, 8.36136715e-16, -1.33226763e-15,
        3.52495810e-15, 1.70349845e-15, -1.75900960e-15, -1.94462502e-15,
        3.67761377e-15, -4.38885039e-16, -2.82412982e-15, -8.67361738e-17,
                -1.43982049e-16, -6.13658430e-17, -7.24420524e-15,  1.00000000e+00,
        2.35367281e-14, 3.21964677e-15, -1.91235916e-14, 8.72233888e-02,
        2.60539493e-01, 4.98950089e-01, 6.55790664e-02, 6.68712647e-02,
        1.95445001e-02, 1.29219835e-03, 1.50218058e-02, 1.18720724e-01,
        1.70893232e-01, 1.03375868e-01, 6.34792441e-02, 2.30980455e-02,
        6.73558391e-02, 9.02923599e-02, 1.87368761e-01, 2.82668390e-02,
        5.15264093e-02, 8.07623970e-03, 1.50218058e-02, 4.84574382e-04,
        8.04393474e-02, 2.40671943e-01, -8.65973959e-15, 4.06557907e-01,
        6.04102730e-02, 2.11920530e-01, 8.08431594e-01, 2.32595703e-02,
        1.22758843e-02, 8.56081408e-03, 1.47472137e-01, 5.47407527e-01,
        4.52592473e-01, 9.51381037e-01, 4.86189630e-02, 9.69633339e-01,
        3.03666613e-02, 1.98675497e-01, 9.36843806e-02, 5.24147957e-01,
        5.68567275e-02, 1.26635439e-01, 9.16653206e-01, 4.84574382e-04,
        3.23049588e-03, 5.16879341e-03, 3.23049588e-03, 2.58439670e-03,
        3.23049588e-04, 4.84574382e-04, 1.93829753e-03, 6.46099176e-04,
        2.42287191e-03, 1.61524794e-03, 1.93829753e-03, 3.23049588e-04,
        6.46099176e-04, 3.23049588e-03, 9.69148764e-04, 1.93829753e-03,
        3.55354547e-03, 1.77677273e-03, 1.29219835e-02, 1.13067356e-03,
        1.13067356e-03, 6.46099176e-04, 2.26134712e-03, 3.23049588e-04,
        8.07623970e-04, 1.61524794e-04, 1.45372315e-03, 2.42287191e-03,
        3.23049588e-04, 1.77677273e-03, 9.69148764e-04, 6.46099176e-04,
        6.46099176e-04, 1.61524794e-04, 2.42287191e-03, 4.84574382e-04,
        1.45372315e-03, 4.84574382e-04, -2.94767466e-18}};
        double[][] tempArray4 = {{ 4.54756381e-02, 1.72853828e-01, 2.48491879e-01, 2.24361949e-01,
        3.08816705e-01, 6.81902552e-01, 1.27842227e-01, 4.38515081e-02,
        2.57540603e-02, 5.17401392e-02, 2.64501160e-02, 6.96055684e-04,
                -1.84314369e-17, 2.01624130e-01, 2.03480278e-01, 2.21113689e-01,
        1.93271462e-01, 1.80510441e-01, 1.11854970e-14, -5.74540415e-15,
                -1.11716192e-15, 1.00000000e+00, -5.30825384e-16, -1.26981758e-15,
        2.91433544e-15, 1.22818422e-15, -1.43288159e-15, -1.34961486e-15,
                -2.40085729e-15, -3.60822483e-16, -1.92207361e-15, 1.07205911e-15,
                -1.40512602e-16, 7.87130777e-17, 5.27355937e-15, 1.00000000e+00,
                -5.74540415e-15, 3.10862447e-15, 7.68829445e-15, 9.98839907e-01,
        1.72084569e-15, -1.60982339e-15, -4.02455846e-16, 1.51267887e-15,
        5.65519853e-16, 1.16009281e-03, 1.53132251e-02, 2.75406032e-01,
        5.17401392e-02, 9.95359629e-02, 9.60556845e-02, 2.08816705e-02,
        4.50116009e-02, 1.07656613e-01, 4.75638051e-02, 5.31322506e-02,
        1.17401392e-01, 1.70870262e-16, 2.83062645e-02, 2.32018561e-04,
        1.74166237e-15, 3.24825986e-03, 9.92807425e-01, -6.99440506e-15,
        3.94431555e-03, -4.80171458e-15, 9.04640371e-01, 1.90255220e-02,
        9.97679814e-03, 5.80046404e-03, 6.05568445e-02, 2.32018562e-04,
        9.99767981e-01, 9.11368910e-01, 8.86310905e-02, 9.50116009e-01,
        4.98839907e-02, 7.51740139e-02, 3.34106729e-02, 5.28770302e-01,
        9.35034803e-02, 2.69141531e-01, 9.34338747e-01, 9.28074246e-04,
        1.85614849e-03, 1.62412993e-03, 2.55220418e-03, 3.01624130e-03,
        2.32018561e-04, 6.96055684e-04, 1.16009281e-03, 1.16009281e-03,
        1.85614849e-03, 1.85614849e-03, 2.78422274e-03, 4.64037123e-04,
        1.24683250e-17, 2.78422274e-03, 2.55220418e-03, 1.85614849e-03,
        9.28074246e-04, 1.16009281e-03, 1.13689095e-02, 6.96055684e-04,
        1.16009281e-03, 4.64037123e-04, 9.28074246e-04, 6.96055684e-04,
        1.16009281e-03, 4.64037123e-04, 9.28074246e-04, 1.16009281e-03,
        6.96055684e-04, 9.28074246e-04, 6.96055684e-04, 2.32018561e-04,
        4.64037123e-04, 4.64037123e-04, 1.39211137e-03, 6.96055684e-04,
        9.28074246e-04, 2.32018561e-04, 8.33480420e-19}};

        Matrix[] centerMatrix = new Matrix[numOfClusters];
        centerMatrix[0] = new Matrix(tempArray1).transpose();
        centerMatrix[1] = new Matrix(tempArray2).transpose();
        centerMatrix[2] = new Matrix(tempArray3).transpose();
        centerMatrix[3] = new Matrix(tempArray4).transpose();
        return centerMatrix;
    }

    private void printSize(Matrix m){
        System.out.println(m.getRowDimension()+" "+m.getColumnDimension());
    }

    void ComputeMEKA() throws CustomException, JepException {
        JavaRDD<String> stringDataRDD = SingleSparkContext.sc.textFile(inputDataPath, numOfClusters);
        JavaRDD<Matrix> dataRDD = getData(stringDataRDD);
//        if(dataRDD.getNumPartitions()>numOfClusters){
//            throw new CustomException("Please select number of cluster more than "
//                    + dataRDD.getNumPartitions());
//        }

        Matrix dataMatrix = getDataMatrixFromRDD(dataRDD);
        JavaRDD<String> stringLabelRDD = SingleSparkContext.sc.textFile(inputLabelPath);
        JavaRDD<Double> labelRDD = getLabel(stringLabelRDD);
        Matrix labelMatrix = getLabelMatrixFromRDD(labelRDD);



        Matrix[] clusterCenters = getKMeansClusterCenters(stringDataRDD);
//        Matrix[] clusterCenters = getTemporaryMatrix();
        Matrix finalDistance = getFinalDataDistanceFromClusters(clusterCenters, dataMatrix);
        Matrix finalIndex = getFinalDataIndex(finalDistance);
        Matrix clusterCentersMatrix = new Matrix(numOfClusters, numOfCols);
        for(int i=0;i<numOfClusters;i++){
            clusterCentersMatrix.setMatrix(i,i,0, numOfCols-1,clusterCenters[i].transpose());
        }

        Matrix squaredDistance = squareDistanceBetweenMatrix(clusterCentersMatrix, clusterCentersMatrix, gamma);
        double[] flattendAra = squaredDistance.getRowPackedCopy();
        Arrays.sort(flattendAra);
        int mm = (int)Math.ceil((numOfClusters*numOfClusters-numOfClusters)*eta);
        double mekaThreshold;
        if(mm>0){
            mekaThreshold = flattendAra[mm-1]   ;
        }
        else{
            mekaThreshold = 0.0;
        }

        ClusterElements[] indexByClusters = new ClusterElements[numOfClusters];
        for(int i=0;i<numOfClusters;i++){
            indexByClusters[i] = new ClusterElements();
        }
        for(int i=0;i<numOfTotalData;i++){
            indexByClusters[(int)finalIndex.get(i,0)].addElement(i);
        }

        int[] rankList = new int[numOfClusters];
        int[] sumRankList = new int[numOfClusters];
        int sumRank = 0;
        for(int i=0; i<numOfClusters; i++){
            rankList[i] = (int)Math.ceil((targetRank*indexByClusters[i].getClusterLength())/(numOfTotalData*1.0));
            sumRankList[i] = sumRank;
            sumRank += rankList[i];
        }
        if(sumRank != targetRank){
            rankList[numOfClusters-1] -= (sumRank - targetRank);
        }

        Matrix[] UList = new Matrix[numOfClusters];
        for (int i=0;i<numOfClusters;i++){
            rankList[i] = min(rankList[i], indexByClusters[i].getClusterLength());
            int m = min(2*rankList[i], indexByClusters[i].getClusterLength());
            Matrix Y = getSubMatrix(dataMatrix, indexByClusters[i].getElements());
            Matrix U = NysTrain(Y, m, rankList[i], gamma);
            UList[i] = U;
        }

//        print(UList[0]);

        Matrix S = new Matrix(targetRank, targetRank);
        for(int i=0;i<targetRank;i++){
            S.set(i,i,1.0);
        }

        for(int i=0;i<numOfClusters;i++){
            for(int j=i+1; j<numOfClusters; j++){
                if(squaredDistance.get(i,j) >= mekaThreshold){
                    List<Integer> randPermi = getRandomPermutation(indexByClusters[i].getClusterLength());
                    int numi = min(4*rankList[i], indexByClusters[i].getClusterLength());
                    List<Integer> randI = randPermi.subList(0, numi);

                    List<Integer> randPermj = getRandomPermutation(indexByClusters[j].getClusterLength());
                    int numj = min(4*rankList[j], indexByClusters[j].getClusterLength());
                    List<Integer> randJ = randPermj.subList(0, numj);

                    Matrix Ui = getSubMatrix(UList[i], randI);
                    Matrix Uj = getSubMatrix(UList[j], randJ);

                    List<Integer> resListI = getElementsByListIndex(indexByClusters[i].getElements(), randI);
                    Matrix Ai = getSubMatrix(dataMatrix, resListI);
                    List<Integer> resListJ = getElementsByListIndex(indexByClusters[j].getElements(), randJ);
                    Matrix Aj = getSubMatrix(dataMatrix, resListJ);

                    Matrix tmpK = squareDistanceBetweenMatrix(Ai, Aj, gamma);
                    Matrix Z1 = (Ui.transpose().times(Ui)).inverse().times(Ui.transpose());
                    Matrix Z2 = Uj.times((Uj.transpose().times(Uj)).inverse());
                    Matrix Z = Z1.times(tmpK.times(Z2));

                    int si = sumRankList[i];
                    int sj = sumRankList[j];
                    int r = Z.getRowDimension();
                    int c = Z.getColumnDimension();
                    S.setMatrix(si, si+r-1, sj, sj+c-1, Z);
                    S.setMatrix(sj, sj+c-1, si, si+r-1, Z.transpose());
                }
            }
        }

        Matrix finalU = new Matrix(numOfTotalData, targetRank);
        int c = 0;
        for(int k=0;k<numOfClusters;k++){
            Matrix U = UList[k];
            int colNum = U.getColumnDimension();
            for(int i=0;i<indexByClusters[k].getClusterLength();i++){
                int index = indexByClusters[k].getElements().get(i);
                finalU.setMatrix(index, index, c, c+colNum-1, U.getMatrix(i,i,0,colNum-1));
            }
            c+= colNum;
        }

//        print(finalU);

//        SingularValueDecomposition svd = new SingularValueDecomposition(S);
        SVDHolder svdHolder = getPythonSVD(S);
        Matrix SU = svdHolder.getU();
        Matrix SS = svdHolder.getS();
        Matrix invSqrtSS = getElementWisePower(SS, -0.5);
        Matrix dataX = finalU.times(SU.times(invSqrtSS));

        List<Integer> randPermData= getRandomPermutation(numOfTotalData);
        int trainDataCount = (int)(numOfTotalData*splitPartition);
        List<Integer> randTrain = randPermData.subList(0, trainDataCount);
        List<Integer> randTest = randPermData.subList(trainDataCount, numOfTotalData);
        trainData = getSubMatrix(dataX, randTrain);
        trainLabel = getSubMatrix(labelMatrix, randTrain);
        testData = getSubMatrix(dataX, randTest);
        testLabel = getSubMatrix(labelMatrix, randTest);
        trainDataWithLabel = elementWiseMultiplication(trainData, trainLabel);
    }
}
