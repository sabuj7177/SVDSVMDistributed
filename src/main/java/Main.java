import Jama.Matrix;
import jep.JepException;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.Accumulator;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import static java.lang.Math.abs;
import static java.lang.Math.floor;

public class Main {
    private static int numOfClusters = 4;
    private static int totalDataSize;
    private static int nCols = 41;
    private static int testRowCount;
    private static int sparkExtraMatrixSize = 10; //For the case when data is not evenly distributed
    private static int partitionCounter = 0;
    private static int partitionCounter2 = 0;
    private static int rowNumberCount = 0;
    private static int clusterCounter = 0;
    private static int fullIterator = 0;
    private static int partialIterator = 0;
    private static int maxIteration = 500;

    private static double learnRate = 0.2;
    private static double C = 1.0;
    private static double testLabelPositive = 1.0; //Positive label of the data set
    private static double testLabelNegative = -1.0; //Negative label of the data set

    private static String datasetName = "a9a";
    private static String variance = "0";
    private static String trainDataPath = "src/main/resources/a9a/a9a_train_data_with_label.txt";
    private static String trainLabelPath = "src/main/resources/a9a/a9a_label_train_col_123.txt";
    private static String testDataPath = "src/main/resources/a9a/a9a_test_after_meka_40.txt";
    private static String testLabelPath = "src/main/resources/a9a/a9a_label_test_col_123.txt";

    private static Matrix globalReflector;
//    private static Matrix recordCountInClustersMat;

    private static Accumulator<Matrix> alphaList;
    private static Accumulator<Matrix> betaList;
    private static Accumulator<Matrix> xAcc;

    private static Broadcast<Matrix> betaBroadcast;
    private static Broadcast<Matrix> enBroadcast;
//    private static JavaRDD<Matrix> localReflectors;
    private static JavaRDD<PerClusterInfo> perClusterInfoJavaRDD;
    private static JavaRDD<PerClusterInfo> perClusterInfoJavaRDD2;

    private static FileWriter fileWriter;

    private static AlphaBetaHolder AlphaBeta(Matrix F, Matrix betaCapOld, Matrix Ecap, int recordCount) {
        Matrix B = Ecap.minus(betaCapOld);
        Matrix alphaCap = new Matrix(recordCount,1);
        alphaCap.setMatrix(0,nCols-1,0, 0, F.inverse().times(B.getMatrix(0,nCols-1,0,0)));
        alphaCap.setMatrix(nCols, recordCount-1,0,0,B.getMatrix(nCols, recordCount-1,0,0).times(-2*C));
//        Matrix alphaCap = (F.inverse()).times();
        double optStepSize = (floor(1.0/(learnRate*C))-0.5)*learnRate;
        Matrix betaCap = betaCapOld.minus(alphaCap.times(optStepSize));
        AlphaBetaHolder holder = new AlphaBetaHolder();
        holder.setAlpha(alphaCap);
        holder.setBeta(betaCap);
        return holder;
    }

    private static AlphaBetaHolder AlphaBetaForOtherCluster(Matrix betaCapOld, Matrix Ecap) {
        Matrix alphaCap = (Ecap.minus(betaCapOld)).times(-2*C);
        double optStepSize = (floor(1.0/(learnRate*C))-0.5)*learnRate;
        Matrix betaCap = betaCapOld.minus(alphaCap.times(optStepSize));
        AlphaBetaHolder holder = new AlphaBetaHolder();
        holder.setAlpha(alphaCap);
        holder.setBeta(betaCap);
        return holder;
    }

    public static void print(Matrix m) {
        for (int i = 0; i < m.getRowDimension(); i++) {
            for (int j = 0; j < m.getColumnDimension(); j++) {
                System.out.print(m.get(i, j) + " ");
            }
            System.out.println();
        }
    }

    public static void print(Matrix m, int row, int column) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                System.out.print(m.get(i, j) + " ");
            }
            System.out.println();
        }
    }

    public static void print(Matrix m, int startRow, int endRow, int startCol, int endCol){
        if(startRow==-1){
            startRow = 0;
        }

        if(endRow == -1){
            endRow = m.getRowDimension();
        }

        if(startCol == -1){
            startCol = 0;
        }

        if(endCol == -1){
            endCol = m.getColumnDimension();
        }

        for (int i = startRow; i < endRow; i++) {
            for (int j = startCol; j < endCol; j++) {
                System.out.print(m.get(i, j) + " ");
            }
            System.out.println();
        }
    }

    public static QRHolder HouseHolderDecomposition2(double[][] A) {
        int m = A.length; // rows
        int n = A[0].length; // cols
        Matrix[] qv = new Matrix[m];
        Matrix[] eara = new Matrix[m];

        Matrix mainMat = new Matrix(A);
        Matrix z = new Matrix(A);
        Matrix z1;

        for (int k = 0; k < n && k < m - 1; k++) {
            Matrix e = new Matrix(new double[m][1]);
            Matrix x = new Matrix(new double[m][1]);
            //Compute Minor
            z1 = new Matrix(m, n);
            for (int i = 0; i < k; i++) {
                z1.set(i, i, 1.0);
            }
            for (int i = k; i < m; i++) {
                for (int j = k; j < n; j++) {
                    z1.set(i, j, z.get(i, j));
                }
            }

            //Extract kth column into x
            for (int i = 0; i < m; i++) {
                x.set(i, 0, z1.get(i, k));
            }

            double a = x.normF();
            if (A[k][k] < 0) {
                a = -a;
            }

            e.set(k, 0, 1.0);

            //Calculate e = x + a*e
            for (int i = 0; i < m; i++) {
                e.set(i, 0, (x.get(i, 0) + (a * e.get(i, 0))));
            }

            //Calculate e = e / ||e||
            double e_val = e.normF();
            for (int i = 0; i < m; i++) {
                e.set(i, 0, e.get(i, 0) / (e_val * 1.0));
            }

            eara[k] = new Matrix(m, 1);
            eara[k].setMatrix(0, m - 1, 0, 0, e);

            // qv[k] = I - 2 *e*e^T
            qv[k] = new Matrix(m, m);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    if (i == j) {
                        qv[k].set(i, j, (1.0 - 2.0 * e.get(i, 0) * e.get(j, 0)));
                    } else {
                        qv[k].set(i, j, (-2.0 * e.get(i, 0) * e.get(j, 0)));
                    }
                }
            }

            // z = qv[k] * z1
            z = qv[k].times(z1);
        }

        Matrix Q = new Matrix(m, m);
        Q.setMatrix(0, m - 1, 0, m - 1, qv[0]);

        // after this loop, we will obtain Q (up to a transpose operation)
        for (int i = 1; i < n && i < m - 1; i++) {
            Matrix temp = qv[i].times(Q);
            Q.setMatrix(0, m - 1, 0, m - 1, temp);
        }
        Matrix R = Q.times(mainMat);
        Q = Q.transpose();

        QRHolder qrh = new QRHolder();
        qrh.setQ(Q);
        qrh.setR(R);

        Matrix total_reflector = new Matrix(m, n);
        for (int i = 0; i < n; i++) {
            total_reflector.setMatrix(0, m - 1, i, i, eara[i]);

        }
        qrh.setReflector(total_reflector);
        return qrh;
    }

    private static QRHolder HouseHolderDecomposition(Matrix A){
        int n = A.getRowDimension();
        int d = A.getColumnDimension();
        Matrix updatedA = new Matrix(A.getArray());
        Matrix reflectors = new Matrix(n,d);

        for(int i=0;i<d;i++){
            double norm_val = updatedA.getMatrix(i, n-1, i, i).normF();
            reflectors.setMatrix(i, n-1, i, i, updatedA.getMatrix(i, n-1, i, i));
            double updated_val = reflectors.get(i, i) + norm_val*(updatedA.get(i,i)<0?-1:1);
            reflectors.set(i, i, updated_val);
            Matrix col_mat = reflectors.getMatrix(i, n-1, i, i).times(1.0/(reflectors.getMatrix(i, n-1, i, i).normF()));
            reflectors.setMatrix(i, n-1, i, i, col_mat);

            Matrix minus_result = updatedA.getMatrix(i, n-1, i, d-1).minus(col_mat.times(col_mat.transpose())
                    .times(updatedA.getMatrix(i, n-1, i, d-1)).times(2));
            updatedA.setMatrix(i, n-1, i, d-1, minus_result);
            for(int j=i;j<n;j++){
                for(int k=i;k<d;k++){
                    if(abs(updatedA.get(j, k))<0.0001){
                        updatedA.set(j, k, 0.0);
                    }
                }
            }

        }

        QRHolder qrh = new QRHolder();
        qrh.setReflector(reflectors);
        qrh.setR(updatedA.getMatrix(0, d-1, 0, d-1));
        return qrh;
    }

    private static Matrix LocalQtX(Matrix reflector, Matrix subX){
        int dataInCluster = subX.getRowDimension();
        for (int k = 0; k < nCols; k++) {
            double value = ((reflector.getMatrix(0, reflector.getRowDimension() - 1, k, k)).transpose()
                    .times(subX)).get(0, 0);
            Matrix temp2 = reflector.getMatrix(0, reflector.getRowDimension() - 1, k, k).times(2 * value);
            subX.setMatrix(0, dataInCluster - 1, 0, 0, subX.minus(temp2));
        }
        return subX;
    }

    private static Matrix GlobalQtX(Matrix partialX){
        for (int k = 0; k < nCols; k++) {
            double value = ((globalReflector.getMatrix(0, globalReflector.getRowDimension() - 1, k, k)).transpose()
                    .times(partialX)).get(0, 0);
            Matrix temp2 = globalReflector.getMatrix(0, globalReflector.getRowDimension() - 1, k, k).times(2 * value);
            partialX.setMatrix(0, numOfClusters * nCols - 1, 0, 0, partialX.minus(temp2));
        }
        return partialX;
    }

    private static void DistributedQtX(boolean betaCalculate, boolean eCalculate) {
//        partitionCounter = 0;
        xAcc.setValue(new Matrix(numOfClusters*nCols,1));
        perClusterInfoJavaRDD = perClusterInfoJavaRDD2.mapPartitions(clusterIterator -> {
            PerClusterInfo clusterInfo = clusterIterator.next();
//            System.out.println("Inside distributed qtx first "+clusterInfo.getClusterNumber());
            Matrix temp = new Matrix(numOfClusters*nCols, 1);
            if(eCalculate){
                Matrix updatedE = LocalQtX(clusterInfo.getLocalReflector(), clusterInfo.getE());
                clusterInfo.setE(updatedE);
                temp.setMatrix(clusterInfo.getClusterNumber()*nCols, (clusterInfo.getClusterNumber()+1)*nCols-1, 0, 0, updatedE.getMatrix(0,nCols-1,0,0));
            }
            if(betaCalculate){
                Matrix updatedBeta = LocalQtX(clusterInfo.getLocalReflector(), clusterInfo.getBeta());
                clusterInfo.setBeta(updatedBeta);
                temp.setMatrix(clusterInfo.getClusterNumber()*nCols, (clusterInfo.getClusterNumber()+1)*nCols-1, 0, 0, updatedBeta.getMatrix(0,nCols-1,0,0));
            }
            xAcc.add(temp);
//            partitionCounter += 1;
            List<PerClusterInfo> new_List = new ArrayList();
            new_List.add(clusterInfo);
            return new_List.iterator();
        }).persist(StorageLevel.MEMORY_ONLY_SER());

        JavaRDD<Integer> dummyMap = perClusterInfoJavaRDD.map((Function<PerClusterInfo, Integer>) s -> 0);
        dummyMap.reduce((Function2<Integer, Integer, Integer>) Integer::sum);
        perClusterInfoJavaRDD2.unpersist();

        Matrix updatedPartialX = GlobalQtX(xAcc.value());

//        partitionCounter = 0;
        perClusterInfoJavaRDD2 = perClusterInfoJavaRDD.mapPartitions(clusterIterator -> {
            PerClusterInfo clusterInfo = clusterIterator.next();
//            System.out.println("Inside distributed qtx second "+clusterInfo.getClusterNumber());
            if(eCalculate){
                clusterInfo.setPartialE(0,nCols-1,0,0,updatedPartialX.getMatrix(clusterInfo.getClusterNumber()*nCols,(clusterInfo.getClusterNumber()+1)*nCols-1,0,0));
            }
            if(betaCalculate){
                clusterInfo.setPartialBeta(0,nCols-1,0,0,updatedPartialX.getMatrix(clusterInfo.getClusterNumber()*nCols,(clusterInfo.getClusterNumber()+1)*nCols-1,0,0));
            }
//            partitionCounter += 1;
            List<PerClusterInfo> new_List = new ArrayList();
            new_List.add(clusterInfo);
            return new_List.iterator();
        }).persist(StorageLevel.MEMORY_ONLY_SER());

        dummyMap = perClusterInfoJavaRDD2.map((Function<PerClusterInfo, Integer>) s -> 0);
        dummyMap.reduce((Function2<Integer, Integer, Integer>) Integer::sum);
        perClusterInfoJavaRDD.unpersist();
    }

//    private static Matrix Dist_QtX(Matrix x){
//        xAcc.setValue(new Matrix(totalDataSize,1));
//        fullIterator = 0;
//        localReflectors.foreachPartition((VoidFunction<Iterator<Matrix>>) arg0 -> {
//            Matrix reflector = arg0.next();
//            int rowIncluster = reflector.getRowDimension();
//            Matrix subX = x.getMatrix(fullIterator,fullIterator+rowIncluster-1,0,0);
//            Matrix updatedSubX = LocalQtX(reflector,subX);
//            Matrix temp = new Matrix(totalDataSize, 1);
//            temp.setMatrix(fullIterator, fullIterator + rowIncluster - 1, 0, 0, updatedSubX);
//            xAcc.add(temp);
//            fullIterator += rowIncluster;
//        });
//
//        Matrix finalX = xAcc.value();
//        Matrix partialX = new Matrix(numOfClusters*nCols,1);
//        fullIterator = 0;
//        partialIterator = 0;
//        for(int k=0;k<numOfClusters;k++){
//            Matrix temp = finalX.getMatrix(fullIterator,fullIterator+nCols-1,0,0);
//            partialX.setMatrix(partialIterator,partialIterator+nCols-1,0,0,temp);
//            fullIterator += (int) recordCountInClustersMat.get(k, 0);
//            partialIterator += nCols;
//        }
//        Matrix updatedPartialX = GlobalQtX(partialX);
//        rowNumberCount = 0;
//        partialIterator = 0;
//        for (int k = 0; k < numOfClusters; k++) {
//            finalX.setMatrix(rowNumberCount, rowNumberCount + nCols - 1, 0, 0, updatedPartialX.getMatrix(partialIterator, partialIterator + nCols - 1, 0, 0));
//            rowNumberCount += (int) recordCountInClustersMat.get(k, 0);
//            partialIterator += nCols;
//        }
//        return finalX;
//    }

    private static Matrix LocalQX(Matrix reflector, Matrix subX){
        int rowIncluster = subX.getRowDimension();
        for (int k = nCols - 1; k >= 0; k--) {
            double value = ((reflector.getMatrix(0, reflector.getRowDimension() - 1, k, k)).transpose()
                    .times(subX)).get(0, 0);
            Matrix temp2 = reflector.getMatrix(0, reflector.getRowDimension() - 1, k, k).times(2 * value);
            subX.setMatrix(0, rowIncluster - 1, 0, 0, subX.minus(temp2));
        }
        return subX;
    }

    private static Matrix GlobalQX(Matrix partialX){
        for (int k = nCols - 1; k >= 0; k--) {
            double value = ((globalReflector.getMatrix(0, globalReflector.getRowDimension() - 1, k, k)).transpose()
                    .times(partialX)).get(0, 0);
            Matrix temp2 = globalReflector.getMatrix(0, globalReflector.getRowDimension() - 1, k, k).times(2 * value);
            partialX.setMatrix(0, numOfClusters * nCols - 1, 0, 0, partialX.minus(temp2));
        }
        return partialX;
    }

    private static void DistributedQX() {
//        partitionCounter2 = 0;
        xAcc.setValue(new Matrix(numOfClusters*nCols,1));
//        System.out.println(partitionCounter2);
        perClusterInfoJavaRDD2 = perClusterInfoJavaRDD.mapPartitions(clusterIterator -> {
//            System.out.println(partitionCounter2);
            PerClusterInfo clusterInfo = clusterIterator.next();
            Matrix temp = new Matrix(numOfClusters*nCols, 1);
            Matrix tt = clusterInfo.getBeta().getMatrix(0,nCols-1,0,0);
//            System.out.println(partitionCounter*nCols+" "+((partitionCounter+1)*nCols-1));
            temp.setMatrix(clusterInfo.getClusterNumber()*nCols, (clusterInfo.getClusterNumber()+1)*nCols-1, 0, 0, clusterInfo.getBeta().getMatrix(0,nCols-1,0,0));
            xAcc.add(temp);
//            partitionCounter2 += 1;
            List<PerClusterInfo> new_List = new ArrayList();
            new_List.add(clusterInfo);
            return new_List.iterator();
        }).persist(StorageLevel.MEMORY_ONLY_SER());

        JavaRDD<Integer> dummyMap = perClusterInfoJavaRDD2.map((Function<PerClusterInfo, Integer>) s -> 0);
        dummyMap.reduce((Function2<Integer, Integer, Integer>) Integer::sum);
        perClusterInfoJavaRDD.unpersist();

        Matrix updatedPartialX = GlobalQX(xAcc.value());

//        partitionCounter2 = 0;
        perClusterInfoJavaRDD = perClusterInfoJavaRDD2.mapPartitions(clusterIterator -> {
            PerClusterInfo clusterInfo = clusterIterator.next();
            Matrix beta = clusterInfo.getBeta();
            beta.setMatrix(0,nCols-1,0,0,updatedPartialX.getMatrix(clusterInfo.getClusterNumber()*nCols,(clusterInfo.getClusterNumber()+1)*nCols-1,0,0));
            Matrix updatedBeta = LocalQX(clusterInfo.getLocalReflector(),beta);
            clusterInfo.setBeta(updatedBeta);
//            partitionCounter2 += 1;
            List<PerClusterInfo> new_List = new ArrayList();
            new_List.add(clusterInfo);
            return new_List.iterator();
        }).persist(StorageLevel.MEMORY_ONLY_SER());

        dummyMap = perClusterInfoJavaRDD.map((Function<PerClusterInfo, Integer>) s -> 0);
        dummyMap.reduce((Function2<Integer, Integer, Integer>) Integer::sum);
        perClusterInfoJavaRDD2.unpersist();
    }

//    private static Matrix Dist_QX(Matrix x){
//        Matrix partialX = new Matrix(numOfClusters*nCols,1);
//        fullIterator = 0;
//        partialIterator = 0;
//        for(int k=0;k<numOfClusters;k++){
//            Matrix temp = x.getMatrix(fullIterator,fullIterator+nCols-1,0,0);
//            partialX.setMatrix(partialIterator,partialIterator+nCols-1,0,0,temp);
//            fullIterator += (int) recordCountInClustersMat.get(k, 0);
//            partialIterator += nCols;
//        }
//        Matrix updatedPartialX = GlobalQX(partialX);
//        rowNumberCount = 0;
//        partialIterator = 0;
//        for (int k = 0; k < numOfClusters; k++) {
//            x.setMatrix(rowNumberCount, rowNumberCount + nCols - 1, 0, 0, updatedPartialX.getMatrix(partialIterator, partialIterator + nCols - 1, 0, 0));
//            rowNumberCount += (int) recordCountInClustersMat.get(k, 0);
//            partialIterator += nCols;
//        }
//
//        xAcc.setValue(new Matrix(totalDataSize,1));
//        fullIterator = 0;
//        localReflectors.foreachPartition((VoidFunction<Iterator<Matrix>>) arg0 -> {
//            Matrix reflector = arg0.next();
//            int rowIncluster = reflector.getRowDimension();
//            Matrix subX = x.getMatrix(fullIterator,fullIterator+rowIncluster-1,0,0);
//            Matrix updatedSubX = LocalQX(reflector,subX);
//            Matrix temp = new Matrix(totalDataSize, 1);
//            temp.setMatrix(fullIterator, fullIterator + rowIncluster - 1, 0, 0, updatedSubX);
//            xAcc.add(temp);
//            fullIterator += rowIncluster;
//        });
//
//        return xAcc.value();
//    }

//    private static Matrix Dist_QX2(Matrix tempx){
//        Matrix x = new Matrix(tempx.getRowDimension(), tempx.getColumnDimension());
//        for(int i=0;i<tempx.getRowDimension();i++){
//            for(int j=0;j<tempx.getColumnDimension();j++){
//                x.set(i,j,tempx.get(i,j));
//            }
//        }
//
//        Matrix partialX = new Matrix(numOfClusters*nCols,1);
//        fullIterator = 0;
//        partialIterator = 0;
//        for(int k=0;k<numOfClusters;k++){
//            Matrix temp = x.getMatrix(fullIterator,fullIterator+nCols-1,0,0);
//            partialX.setMatrix(partialIterator,partialIterator+nCols-1,0,0,temp);
//            fullIterator += (int) recordCountInClustersMat.get(k, 0);
//            partialIterator += nCols;
//        }
//        Matrix updatedPartialX = GlobalQX(partialX);
//        rowNumberCount = 0;
//        partialIterator = 0;
//        for (int k = 0; k < numOfClusters; k++) {
//            x.setMatrix(rowNumberCount, rowNumberCount + nCols - 1, 0, 0, updatedPartialX.getMatrix(partialIterator, partialIterator + nCols - 1, 0, 0));
//            rowNumberCount += (int) recordCountInClustersMat.get(k, 0);
//            partialIterator += nCols;
//        }
//
//        xAcc.setValue(new Matrix(totalDataSize,1));
//        fullIterator = 0;
//        localReflectors.foreachPartition((VoidFunction<Iterator<Matrix>>) arg0 -> {
//            Matrix reflector = arg0.next();
//            int rowIncluster = reflector.getRowDimension();
//            Matrix subX = x.getMatrix(fullIterator,fullIterator+rowIncluster-1,0,0);
//            Matrix updatedSubX = LocalQX(reflector,subX);
//            Matrix temp = new Matrix(totalDataSize, 1);
//            temp.setMatrix(fullIterator, fullIterator + rowIncluster - 1, 0, 0, updatedSubX);
//            xAcc.add(temp);
//            fullIterator += rowIncluster;
//        });
//
//        return xAcc.value();
////        return new Matrix(totalDataSize,1);
//    }

    private static void TestData(Matrix alphaTotalMatrix, Matrix finalR, Matrix testDataMatrix, Matrix testLabelMatrix, double bias) throws IOException {
        Matrix weightMatrix = (finalR.transpose()).times(alphaTotalMatrix.getMatrix(0, nCols - 1, 0, 0));
//        weightMatrix.set(nCols-1,0,bias);
        Matrix transposeWeightMatrix = weightMatrix.transpose();
        Matrix result = transposeWeightMatrix.times(testDataMatrix.transpose());
        int correct = 0;
        int wrong = 0;
        int correctPos = 0;
        int correctNeg = 0;
        int wrongPos = 0;
        int wrongNeg = 0;
        for (int i = 0; i < testRowCount; i++) {
            if (result.get(0, i) > 0) {
                if (testLabelMatrix.get(i, 0) == testLabelPositive) {
                    correct += 1;
                    correctPos += 1;
                } else {
                    wrong += 1;
                    wrongPos += 1;
                }
            } else {
                if (testLabelMatrix.get(i, 0) == testLabelNegative) {
                    correct += 1;
                    correctNeg += 1;
                } else {
                    wrong += 1;
                    wrongNeg += 1;
                }
            }
        }
        System.out.println("Correct is " + correct);
        System.out.println("Wrong is " + wrong);
        double accuracy = (correct * 100.0) / testRowCount;
        System.out.println("Accuracy is " + accuracy);
        System.out.println("Correct positive is " + correctPos);
        System.out.println("Correct negative is " + correctNeg);
        System.out.println("Wrong positive is " + wrongPos);
        System.out.println("Wrong negative is " + wrongNeg);

        fileWriter.write("Correct is " + correct + "\n");
        fileWriter.write("Wrong is " + wrong + "\n");
        fileWriter.write("Accuracy is " + accuracy + "\n");
        fileWriter.write("Correct positive is " + correctPos + "\n");
        fileWriter.write("Correct negative is " + correctNeg + "\n");
        fileWriter.write("Wrong positive is " + wrongPos + "\n");
        fileWriter.write("Wrong negative is " + wrongNeg + "\n");
        fileWriter.flush();
    }

    private static List<Matrix> getRowListFromMatrix(Matrix m){
        int rowLen = m.getRowDimension();
        int colLen = m.getColumnDimension();
        List<Matrix> rowList = new ArrayList<>();
        for(int i=0;i<rowLen;i++){
            rowList.add(m.getMatrix(i,i,0,colLen-1));
        }
        return rowList;
    }

//    private static double getBias(Matrix trainDataWithoutLabel, Matrix trainLabel, Matrix alphaMat){
//        Matrix supportValues = Dist_QX2(alphaMat);
////        Matrix supportValues = alphaMat;
//        List<Integer> supportValuesIndexList = new ArrayList<>();
//        List<Double> supportValuesList = new ArrayList<>();
//        for(int i=0;i<totalDataSize;i++){
//            if (supportValues.get(i,0)>0.001){
//                supportValuesIndexList.add(i);
//                supportValuesList.add(supportValues.get(i,0));
//            }
//        }
//
//        Matrix Xk = trainDataWithoutLabel.getMatrix(supportValuesIndexList.get(0), supportValuesIndexList.get(0), 0, nCols-1);
//        int svCount = supportValuesIndexList.size();
//        double biasLocal = 0;
//        for(int i=0;i<svCount;i++){
//            Matrix temp = trainDataWithoutLabel.getMatrix(supportValuesIndexList.get(i), supportValuesIndexList.get(i), 0, nCols-1);
//            double scalerValue = temp.times(Xk.transpose()).get(0,0);
//            biasLocal += supportValuesList.get(i)*trainLabel.get(supportValuesIndexList.get(i), 0)*scalerValue;
//        }
//
//        double bias = trainLabel.get(supportValuesIndexList.get(0), 0) - biasLocal;
//        return bias;
////        return 0.1;
//    }

    public static void main(String[] args) throws IOException, CustomException, JepException {
        // Initialization
//        String absolutePath = "C:\\Jep\\jep\\build\\lib.win-amd64-3.6\\jep\\jep.cp36-win_amd64.dll";
//        MainInterpreter.setJepLibraryPath(absolutePath);
        xAcc = SingleSparkContext.sc.accumulator(new Matrix(totalDataSize, 1), new MatrixAccumulatorParam());
        String filePathPrefix = "D:\\L4T1\\Thesis\\thesis codes\\SVDSVMDistributed\\SVDSVMDistributed\\src\\main\\resources\\";
        fileWriter = new FileWriter(filePathPrefix + datasetName + "\\logfile_"+datasetName+"_"+variance+"_"+nCols+".txt");
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);

//        Instant mekaStart = Instant.now();
        // Memory Efficient Kernel Approximation
//        MEKAHolder mekaHolder = new MEKAHolder("src/main/resources/a9a/a9a_data_for_meka_col_123.txt",
//                "src/main/resources/a9a/a9a_label_for_meka_col_123.txt",
//                4, 32561, 123, 10, nCols, 0.1, 0.1, 0.8);
//        mekaHolder.ComputeMEKA();
//        Matrix trainDataWithLabel = mekaHolder.getTrainDataWithLabel();
//        Matrix testData = mekaHolder.getTestData();
//        Matrix testLabel = mekaHolder.getTestLabel();
//        testRowCount = testData.getRowDimension();
//        totalDataSize = trainDataWithLabel.getRowDimension();
        Instant mekaEnd = Instant.now();

        /*try {
            trainDataPath = System.getProperty("trainData");
            if (trainDataPath == null)
                throw new IllegalArgumentException();
        } catch (Exception e) {
            System.out.println("Train Data Path is Missing");
            return;
        }
        try {
            testDataPath = System.getProperty("testData");
            if (testDataPath == null)
                throw new IllegalArgumentException();
        } catch (Exception e) {
            System.out.println("Test Data Path is Missing");
            return;
        }

        try {
            testLabelPath = System.getProperty("testLabel");
            if (trainDataPath == null)
                throw new IllegalArgumentException();
        } catch (Exception e) {
            System.out.println("Test Label Path is Missing");
            return;
        }

        try {
            nRows = Integer.parseInt(System.getProperty("trainRows"));
        } catch (Exception e) {
            System.out.println("Train Rows is Missing");
            return;
        }

        try {
            nCols = Integer.parseInt(System.getProperty("cols"));
        } catch (Exception e) {
            System.out.println("nCols is Missing");
            return;
        }

        try {
            testRowCount = Integer.parseInt(System.getProperty("testRows"));
        } catch (Exception e) {
            System.out.println("Test Rows is Missing");
            return;
        }

        try {
            C = Integer.parseInt(System.getProperty("c"));
        } catch (Exception e) {
            System.out.println("C is Missing");
            return;
        }

        try {
            eta = Integer.parseInt(System.getProperty("eta"));
        } catch (Exception e) {
            System.out.println("Eta is Missing");
            return;
        }

        try {
            powerIteration = Integer.parseInt(System.getProperty("powit"));
        } catch (Exception e) {
            System.out.println("Power Iteration is Missing");
            return;
        }

        try {
            nClusters = Integer.parseInt(System.getProperty("nClusters"));
        } catch (Exception e) {
            System.out.println("nClusters is Missing");
            return;
        }

        try {
            targetDimension = Integer.parseInt(System.getProperty("td"));
        } catch (Exception e) {
            System.out.println("Target Dimension is Missing");
            return;
        }*/


        totalDataSize = 26048;
        testRowCount = 6513;

        JavaRDD<Matrix> trainingDataRDD = SingleSparkContext.sc.textFile(trainDataPath, numOfClusters)
                .map((Function<String, Matrix>) s -> {
                    String[] sarray = s.trim().split(" ");
                    double[][] values = new double[1][nCols];
                    for (int i = 0; i < nCols; i++) {
                        values[0][i] = Double.parseDouble(sarray[i]);
                    }
                    return new Matrix(values);
                }
                ).persist(StorageLevel.MEMORY_ONLY_SER());

        // When reading data from text file, generally data count in all clusters does not become same. So we need an
        // accumulator to store which clusters have how many records.
//        Accumulator<Matrix> recordCountInClusters = SingleSparkContext.sc.accumulator(new Matrix(numOfClusters, 1), new MatrixAccumulatorParam());

        //Accumulator to gather partial R
        Accumulator<Matrix> rGatherAcc = SingleSparkContext.sc.accumulator(new Matrix(numOfClusters * nCols, nCols), new MatrixAccumulatorParam());
        partitionCounter = 0;
        perClusterInfoJavaRDD = trainingDataRDD.mapPartitions(matrixIterator -> {
            //As data is not evenly distributed, we are making matrix with slightly larger size.
            Matrix partitionedMatrix = new Matrix((totalDataSize / numOfClusters) + sparkExtraMatrixSize, nCols);
            int currentRow = 0;
            while (matrixIterator.hasNext()) {
                Matrix rowMat = matrixIterator.next();
                double label = rowMat.get(0,nCols-1);
                rowMat.set(0,nCols-1,1.0);
                partitionedMatrix.setMatrix(currentRow, currentRow, 0, nCols - 1, rowMat.times(label));
                currentRow++;
            }

            //We are trimming rows which are not used from partitionedMatrix.
            Matrix trimmedMatrix = partitionedMatrix.getMatrix(0, currentRow - 1, 0, nCols - 1);

            // Generating R matrix
            QRHolder qrd = HouseHolderDecomposition(trimmedMatrix);
            Matrix temp = new Matrix(numOfClusters * nCols, nCols);
            temp.setMatrix(partitionCounter * nCols, (partitionCounter+1) * nCols - 1, 0, nCols - 1, qrd.getR());
            rGatherAcc.add(temp);

//            // Generating matrix for record count in each clusters.
//            Matrix temp2 = new Matrix(numOfClusters, 1);
//            temp2.set(partitionCounter, 0, currentRow);
//            recordCountInClusters.add(temp2);
//            System.out.println("Inside training data rdd " + partitionCounter);

            //Returning generated Reflector from each cluster to a RDD
            System.out.println("Gathering partial reflectors in cluster "+partitionCounter);

            PerClusterInfo perClusterInfo = new PerClusterInfo();
            perClusterInfo.setLocalReflector(qrd.getReflector());
            perClusterInfo.setDataCount(currentRow);
            perClusterInfo.setClusterNumber(partitionCounter);
            partitionCounter += 1;
            List<PerClusterInfo> new_List = new ArrayList();
            new_List.add(perClusterInfo);
            return new_List.iterator();
        }).persist(StorageLevel.MEMORY_ONLY_SER());

        //Spark uses lazy waiting for mapPartitions method. So we are using this codes to compute QReflectors early
        JavaRDD<Integer> dummyMap = perClusterInfoJavaRDD.map((Function<PerClusterInfo, Integer>) s -> 0);
        dummyMap.reduce((Function2<Integer, Integer, Integer>) Integer::sum);

        //Generating final R and final R inverse.
        Matrix rGatherMat = rGatherAcc.value();
        QRHolder qrHolder = HouseHolderDecomposition(rGatherMat);
        globalReflector = qrHolder.getReflector();
        Matrix finalR = qrHolder.getR();
//        recordCountInClustersMat = recordCountInClusters.value();
        //Creating RgRgT
        Matrix RgRgTmat = finalR.times(finalR.transpose());
        System.out.println("QR decomposition complete");

//        partitionCounter = 0;
        perClusterInfoJavaRDD2 = perClusterInfoJavaRDD.mapPartitions(clusterIterator -> {
            PerClusterInfo clusterInfo = clusterIterator.next();
//            System.out.println("Inside creation of F "+clusterInfo.getClusterNumber());
            if(clusterInfo.getClusterNumber()==0){
                Matrix F = new Matrix(nCols, nCols);
                for (int i = 0; i < nCols; i++) {
                    for (int j = 0; j < nCols; j++) {
                        F.set(i, j, (RgRgTmat.get(i,j) * (-1)));
                    }
                }

                for (int i = 0; i < nCols; i++) {
                    double dval = F.get(i, i);
                    dval += (-1.0 / (2.0 * C));
                    F.set(i, i, dval);
                }
                clusterInfo.setF(F);
            }

            clusterInfo.setBeta(new Matrix(clusterInfo.getDataCount(),1,1.0));
            clusterInfo.setPrevBeta(new Matrix(clusterInfo.getDataCount(),1,1.0));
            clusterInfo.setE(new Matrix(clusterInfo.getDataCount(),1,-1.0));
//            partitionCounter += 1;
            List<PerClusterInfo> new_List = new ArrayList();
            new_List.add(clusterInfo);
            return new_List.iterator();
        }).persist(StorageLevel.MEMORY_ONLY_SER());

//        perClusterInfoJavaRDD.foreachPartition(clusterIterator -> {
//            PerClusterInfo clusterInfo = clusterIterator.next();
//
//            print(clusterInfo.getE());
////            List<PerClusterInfo> new_List = new ArrayList();
////            new_List.add(clusterInfo);
////            return new_List.iterator();
//        });

        dummyMap = perClusterInfoJavaRDD2.map((Function<PerClusterInfo, Integer>) s -> 0);
        dummyMap.reduce((Function2<Integer, Integer, Integer>) Integer::sum);

        perClusterInfoJavaRDD.unpersist();

        DistributedQtX(false, true);

        for(int it = 0;it<maxIteration;it++){
//            partitionCounter = 0;
//            System.out.println("Partition number "+perClusterInfoJavaRDD.getNumPartitions());
            perClusterInfoJavaRDD = perClusterInfoJavaRDD2.mapPartitions(clusterIterator -> {
                PerClusterInfo clusterInfo = clusterIterator.next();
//                System.out.println("Inside alpha beta calculation "+clusterInfo.getClusterNumber());
                AlphaBetaHolder holder;
                if (clusterInfo.getClusterNumber()==0){
                    holder = AlphaBeta(clusterInfo.getF(), clusterInfo.getBeta(), clusterInfo.getE(), clusterInfo.getDataCount());
                }
                else {
                    holder = AlphaBetaForOtherCluster(clusterInfo.getBeta(), clusterInfo.getE());
                }
                clusterInfo.setAlphaCap(holder.getAlpha());
                clusterInfo.setBeta(holder.getBeta());

//                partitionCounter += 1;
                List<PerClusterInfo> new_List = new ArrayList();
                new_List.add(clusterInfo);
                return new_List.iterator();
            }).persist(StorageLevel.MEMORY_ONLY_SER());

            dummyMap = perClusterInfoJavaRDD.map((Function<PerClusterInfo, Integer>) s -> 0);
            dummyMap.reduce((Function2<Integer, Integer, Integer>) Integer::sum);
            perClusterInfoJavaRDD2.unpersist();

            DistributedQX();

            perClusterInfoJavaRDD2 = perClusterInfoJavaRDD.mapPartitions(clusterIterator -> {
                PerClusterInfo clusterInfo = clusterIterator.next();
//                System.out.println("Inside non negativity check "+clusterInfo.getClusterNumber());
                Matrix beta = clusterInfo.getBeta();
                for(int i=0;i<clusterInfo.getDataCount();i++){
                    if(beta.get(i,0)<0){
                        beta.set(i,0,0);
                    }
                }
                clusterInfo.setBeta(beta);
                List<PerClusterInfo> new_List = new ArrayList();
                new_List.add(clusterInfo);
                return new_List.iterator();
            }).persist(StorageLevel.MEMORY_ONLY_SER());

            dummyMap = perClusterInfoJavaRDD2.map((Function<PerClusterInfo, Integer>) s -> 0);
            dummyMap.reduce((Function2<Integer, Integer, Integer>) Integer::sum);
            perClusterInfoJavaRDD.unpersist();

            DistributedQtX(true, false);

//            partitionCounter = 0;
            Accumulator<Matrix> errorAccumulator = SingleSparkContext.sc.accumulator(new Matrix(numOfClusters, 1), new MatrixAccumulatorParam());
            perClusterInfoJavaRDD = perClusterInfoJavaRDD2.mapPartitions(clusterIterator -> {
                PerClusterInfo clusterInfo = clusterIterator.next();
//                System.out.println("Inside error calculation "+clusterInfo.getClusterNumber());
                Matrix diffbeta = clusterInfo.getBeta().minus(clusterInfo.getPrevBeta());
                double error = diffbeta.norm1();
                Matrix temp = new Matrix(numOfClusters,1);
                temp.set(clusterInfo.getClusterNumber(),0,error);
                errorAccumulator.add(temp);
                clusterInfo.copyBetaToPrevBeta(clusterInfo.getBeta());
//                partitionCounter += 1;
                List<PerClusterInfo> new_List = new ArrayList();
                new_List.add(clusterInfo);
                return new_List.iterator();
            }).persist(StorageLevel.MEMORY_ONLY_SER());

            dummyMap = perClusterInfoJavaRDD.map((Function<PerClusterInfo, Integer>) s -> 0);
            dummyMap.reduce((Function2<Integer, Integer, Integer>) Integer::sum);
            perClusterInfoJavaRDD2.unpersist();

            perClusterInfoJavaRDD2 = perClusterInfoJavaRDD.mapPartitions(clusterIterator -> {
                PerClusterInfo clusterInfo = clusterIterator.next();
                List<PerClusterInfo> new_List = new ArrayList();
                new_List.add(clusterInfo);
                return new_List.iterator();
            }).persist(StorageLevel.MEMORY_ONLY_SER());

            dummyMap = perClusterInfoJavaRDD2.map((Function<PerClusterInfo, Integer>) s -> 0);
            dummyMap.reduce((Function2<Integer, Integer, Integer>) Integer::sum);
            perClusterInfoJavaRDD.unpersist();

            Matrix errorMatrix = errorAccumulator.value();
            double totalError = 0;
            for(int i=0;i<numOfClusters;i++){
                totalError+=errorMatrix.get(i,0);
            }
            System.out.println("In iteration "+it);
            System.out.println("Total error " + totalError);
            if (totalError < 0.001) {
                break;
            }
        }




















//        //Initializing Beta matrix in 0th iteration
//        Matrix betaCapmat = new Matrix(totalDataSize,1,1.0);
//        betaBroadcast = SingleSparkContext.sc.broadcast(betaCapmat);
//
//        Matrix Ecap = Dist_QtX(new Matrix(totalDataSize,1,-1.0));
//        enBroadcast = SingleSparkContext.sc.broadcast(Ecap);
//
//        List<Matrix> Fs = new ArrayList<>();
//        //Initializing F matrix
//        Matrix[] F = new Matrix[numOfClusters];
//        F[0] = new Matrix(nCols, nCols);
//        for (int i = 0; i < nCols; i++) {
//            for (int j = 0; j < nCols; j++) {
//                F[0].set(i, j, (RgRgTmat.get(i,j) * (-1)));
//            }
//        }
//
//        for (int i = 0; i < nCols; i++) {
//            double dval = F[0].get(i, i);
//            dval += (-1.0 / (2.0 * C));
//            F[0].set(i, i, dval);
//        }
//        Fs.add(F[0]);
//        //Done creating F1
//
//        //Building F2...P
//        for (int p = 1; p < numOfClusters; p++) {
//            F[p] = new Matrix(nCols, nCols);
////            for (int i = 0; i < (int) recordCountInClustersMat.get(p, 0); i++) {
////                double dval = (-1.0 / (2.0 * C));
////                F[p].set(i, i, dval);
////            }
//            Fs.add(F[p]);
//        }

//        JavaRDD<Matrix> FsRDD = SingleSparkContext.sc.parallelize(Fs, numOfClusters);
//        Matrix prevBetacap = new Matrix(totalDataSize, 1);
//        prevBetacap.setMatrix(0,totalDataSize-1,0,0,betaCapmat);
//        for (int it = 0; it < maxIteration; it++) {
//            System.out.println("In iteration "+it);
//            alphaList = SingleSparkContext.sc.accumulator(new Matrix(totalDataSize, 1), new MatrixAccumulatorParam());
//            betaList = SingleSparkContext.sc.accumulator(new Matrix(totalDataSize, 1), new MatrixAccumulatorParam());
//            partitionCounter = 0;
//            rowNumberCount = 0;
//            clusterCounter = 0;
//            FsRDD.foreachPartition((VoidFunction<Iterator<Matrix>>) arg0 -> {
//                Matrix partialF = arg0.next();
////                int partialFLength = partialF.getRowDimension();
//                int partialFLength = (int)recordCountInClustersMat.get(clusterCounter, 0);
//                Matrix Beta = betaBroadcast.value().getMatrix(partitionCounter, partitionCounter + partialFLength - 1, 0, 0);
//                Matrix E = enBroadcast.value().getMatrix(partitionCounter, partitionCounter + partialFLength - 1, 0, 0);
//                AlphaBetaHolder holder;
//                if (clusterCounter==0){
//                    holder = AlphaBeta(partialF, Beta, E, partialFLength);
//                }
//                else {
//                    holder = AlphaBetaForOtherCluster(Beta, E);
//                }
//
//                Matrix temp = new Matrix(totalDataSize, 1);
//                Matrix temp2 = new Matrix(totalDataSize, 1);
//                temp.setMatrix(partitionCounter, partitionCounter + partialFLength - 1, 0, 0, holder.getAlpha());
//                alphaList.add(temp);
//                temp2.setMatrix(rowNumberCount, rowNumberCount + partialFLength - 1, 0, 0, holder.getBeta());
//                betaList.add(temp2);
//                partitionCounter += partialFLength;
//                rowNumberCount += partialFLength;
//                clusterCounter += 1;
//            });
//
//            // Non negativity check
//            Matrix betaMatrix = Dist_QX(betaList.value());
//            for (int i = 0; i < betaMatrix.getRowDimension(); i++) {
//                for (int j = 0; j < betaMatrix.getColumnDimension(); j++) {
//                    if (betaMatrix.get(i, j) < 0) {
//                        betaMatrix.set(i, j, 0);
//                    }
//                }
//            }
//            betaCapmat = Dist_QtX(betaMatrix);
//
//            Matrix diffbeta = betaCapmat.minus(prevBetacap);
//            double error = diffbeta.norm1();
//            for (int i = 0; i < totalDataSize; i++) {
//                prevBetacap.set(i, 0, betaCapmat.get(i, 0));
//            }
//            System.out.println("Total error " + error);
//            fileWriter.write("############################Here is iteration " + it + " ###############" + "\n");
//            fileWriter.write("this is error " + error + "\n");
//            fileWriter.flush();
//
//            betaBroadcast = SingleSparkContext.sc.broadcast(betaCapmat);
////            Matrix alphaTotMatrix = alphaList.value();
////            double tempBias = getBias(trainDataWithoutLabel, trainLabel, alphaTotMatrix);
////            double tempBias = 0;
////            TestData(alphaTotMatrix,finalR,testData,testLabel,tempBias);
//
//            if (error < 0.001) {
//                break;
//            }
//        }

        File testDataFile = new File(testDataPath);
        Scanner scanner = new Scanner(testDataFile);
        Matrix testData = new Matrix(testRowCount,nCols);
        rowNumberCount = 0;
        while (scanner.hasNextLine()){
            if(rowNumberCount>=testRowCount)
                throw new CustomException("Found test data more than test row count");
            String[] sarray = scanner.nextLine().trim().split(",");
            double[][] values = new double[1][nCols];
            for (int i = 0; i < nCols-1; i++) {
                values[0][i] = Double.parseDouble(sarray[i]);
            }
            values[0][nCols-1] = 1.0;
            testData.setMatrix(rowNumberCount, rowNumberCount, 0, nCols - 1, new Matrix(values));
            rowNumberCount++;
        }

        if(rowNumberCount != testRowCount)
            throw new CustomException("Test data number is not same as test row count");

        File testLabelFile = new File(testLabelPath);
        scanner = new Scanner(testLabelFile);
        Matrix testLabel = new Matrix(testRowCount,1);
        rowNumberCount = 0;
        while (scanner.hasNextLine()){
            if(rowNumberCount >= testRowCount)
                throw new CustomException("Found test label more than test row count");
            String[] sarray = scanner.nextLine().trim().split(" ");
            testLabel.set(rowNumberCount, 0, Double.parseDouble(sarray[0]));
            rowNumberCount++;
        }

        if(rowNumberCount != testRowCount)
            throw new CustomException("Test label number is not same as test row count");

        Matrix alphaTotalMatrix = alphaList.value();
//        double bias = getBias(trainDataWithoutLabel, trainLabel, alphaTotalMatrix);
//        System.out.println("Bias is "+bias);
        TestData(alphaTotalMatrix,finalR,testData,testLabel, 1.0);

        Instant svmEnd = Instant.now();
//        Duration timeElapsedMeka = Duration.between(mekaStart, mekaEnd);
//        System.out.println("Time taken in meka: "+ timeElapsedMeka.toMillis() +" milliseconds");
        Duration timeElapsedSVM = Duration.between(mekaEnd, svmEnd);
        System.out.println("Time taken in SVM: "+ timeElapsedSVM.toMillis() +" milliseconds");

        /*final XYSeriesDemo demo = new XYSeriesDemo("Beta change vs Iteration", new Matrix(betadifara), "Amount of beta change");
        demo.pack();
        RefineryUtilities.centerFrameOnScreen(demo);
        demo.setVisible(true);*/
    }
}
