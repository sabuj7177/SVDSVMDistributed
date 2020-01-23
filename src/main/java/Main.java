import Jama.Matrix;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.function.FlatMapFunction;

import java.io.File;
import java.util.*;
import java.io.IOException;
import org.apache.log4j.Level;
import org.apache.spark.Accumulator;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;
import static java.lang.Math.abs;
import static java.lang.Math.floor;

public class Main {

    private static int numOfClusters = 5;
    //private static int totalData = 4000;
    private static int numOfPartialData;
    private static int testrow = 1000;
    private static int nCols = 40;
    //private static int thresholdSpark = 10; //If in case data is not evenly distributed
    //private static int partitionCounter;
    //private static int rowNumberCount = 0;
    //private static int fullIterator = 0;
    //private static int partialIterator = 0;
//    private static Matrix globalReflector;
//    private static Matrix recordCountInClustersMat;
//    private static Accumulator<Matrix> Rgatherac;
//    private static Accumulator<Matrix> alphaList;
//    private static Accumulator<Matrix> betaList;
//    private static Accumulator<Matrix> xAcc;
//    private static Broadcast<Matrix> betaBroadcast;
//    private static Broadcast<Matrix> enBroadcast;
//    private static JavaRDD<Matrix> localReflectors;

    private static double learnRate = 0.2;
    private static double C = 1.0;
    private static double threshold = 0.001;
    private static int maxIteration = 100;
    private static double optStepSize;
//    private static String trainDataPath = "src/main/resources/rcv1/rcv1_train_4000_200_with_meka.txt";
//    private static String testDataPath = "src/main/resources/rcv1/rcv1_test_1000_200.txt";
//    private static String testLabelPath = "src/main/resources/rcv1/rcv1_test_1000_200_with_label.txt";
    private static double testLabelPositive = 1.0; //Positive label of the data set
    private static double testLabelNegative = -1.0; //Negative label of the data set

    private static AlphaBetaHolder AlphaBeta(Matrix F, Matrix betaCapOld, Matrix Ecap) {
        Matrix alphaCap = (F.inverse()).times(Ecap.minus(betaCapOld));
        Matrix betaCap = betaCapOld.minus(alphaCap.times(optStepSize));

        AlphaBetaHolder holder = new AlphaBetaHolder();
        holder.setAlpha(alphaCap);
        holder.setBeta(betaCap);
        return holder;
    }

    /*public static int BetaChange(Matrix M,Matrix mainMat){
        int count = 0;+

        //double[][] m = M.getArray();
        for(int i=0;i<M.getRowDimension();i++){
            double value = (M.get(i, 0)/mainMat.get(i, 0))*100.0;
            if(value>0.1){
                count++;
            }
        }
        return count;
    }*/

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

    private static QRHolder HouseHolderDecomposition(double[][] A){
        int n = A.length;
        int d = A[0].length;
        Matrix updatedA = new Matrix(A);
        Matrix reflectors = new Matrix(n,d);
        for(int i=0;i<d;i++){
            double norm_val = updatedA.getMatrix(i, n-1, i, i).normF();
            reflectors.setMatrix(i, n-1, i, i, updatedA.getMatrix(i, n-1, i, i));
            if(updatedA.get(i, i)<0){
                norm_val = norm_val*-1;
            }
            double updated_val = reflectors.get(i, i) + norm_val;
            reflectors.set(i, i, updated_val);
            Matrix col_mat = reflectors.getMatrix(i, n-1, i, i);
            double col_norm = col_mat.normF();
            col_mat = col_mat.times(1.0/col_norm);
            reflectors.setMatrix(i, n-1, i, i, col_mat);

            Matrix partTransA = updatedA.getMatrix(i, n-1, i, d-1);
            Matrix temp_mat = col_mat.times(col_mat.transpose()).times(partTransA).times(2);
            Matrix minus_result = partTransA.minus(temp_mat);
            updatedA.setMatrix(i, n-1, i, d-1, minus_result);
            for(int j=i;j<n;j++){
                for(int k=i;k<d;k++){
                    if(abs(updatedA.get(j, k))<0.0001){
                        updatedA.set(j, k, 0.0);
                    }
                }
            }

        }

        Matrix R = updatedA.getMatrix(0, d-1, 0, d-1);
        QRHolder qrh = new QRHolder();
        qrh.setReflector(reflectors);
        qrh.setR(R);
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

//    private static Matrix GlobalQtX(Matrix partialX){
//        for (int k = 0; k < nCols; k++) {
//            double value = ((globalReflector.getMatrix(0, globalReflector.getRowDimension() - 1, k, k)).transpose()
//                    .times(partialX)).get(0, 0);
//            Matrix temp2 = globalReflector.getMatrix(0, globalReflector.getRowDimension() - 1, k, k).times(2 * value);
//            partialX.setMatrix(0, numOfClusters * nCols - 1, 0, 0, partialX.minus(temp2));
//        }
//        return partialX;
//    }

//    private static Matrix Dist_QtX(Matrix x){
//        xAcc.setValue(new Matrix(totalData,1));
//        fullIterator = 0;
//        localReflectors.foreachPartition(new VoidFunction<Iterator<Matrix>>() {
//            public void call(Iterator<Matrix> arg0) throws Exception {
//                Matrix reflector = arg0.next();
//                int rowIncluster = reflector.getRowDimension();
//                Matrix subX = x.getMatrix(fullIterator,fullIterator+rowIncluster-1,0,0);
//                Matrix updatedSubX = LocalQtX(reflector,subX);
//                Matrix temp = new Matrix(totalData, 1);
//                temp.setMatrix(fullIterator, fullIterator + rowIncluster - 1, 0, 0, updatedSubX);
//                xAcc.add(temp);
//                fullIterator += rowIncluster;
//            }
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

//    private static Matrix GlobalQX(Matrix partialX){
//        for (int k = nCols - 1; k >= 0; k--) {
//            double value = ((globalReflector.getMatrix(0, globalReflector.getRowDimension() - 1, k, k)).transpose()
//                    .times(partialX)).get(0, 0);
//            Matrix temp2 = globalReflector.getMatrix(0, globalReflector.getRowDimension() - 1, k, k).times(2 * value);
//            partialX.setMatrix(0, numOfClusters * nCols - 1, 0, 0, partialX.minus(temp2));
//        }
//        return partialX;
//    }

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
//        xAcc.setValue(new Matrix(totalData,1));
//        fullIterator = 0;
//        localReflectors.foreachPartition(new VoidFunction<Iterator<Matrix>>() {
//            public void call(Iterator<Matrix> arg0) throws Exception {
//                Matrix reflector = arg0.next();
//                int rowIncluster = reflector.getRowDimension();
//                Matrix subX = x.getMatrix(fullIterator,fullIterator+rowIncluster-1,0,0);
//                Matrix updatedSubX = LocalQX(reflector,subX);
//                Matrix temp = new Matrix(totalData, 1);
//                temp.setMatrix(fullIterator, fullIterator + rowIncluster - 1, 0, 0, updatedSubX);
//                xAcc.add(temp);
//                fullIterator += rowIncluster;
//            }
//        });
//
//        return xAcc.value();
//    }

    private static void TestData(Matrix alphatotalmat, Matrix finalR, Matrix testdatamat, Matrix testlabelmat){
        Matrix weightmat = (finalR.transpose()).times(alphatotalmat.getMatrix(0, nCols - 1, 0, 0));
        Matrix transposeweightmat = weightmat.transpose();
        Matrix result = transposeweightmat.times(testdatamat.transpose());
        int correct = 0;
        int wrong = 0;
        int correctPos = 0;
        int correctNeg = 0;
        int wrongPos = 0;
        int wrongNeg = 0;
        for (int i = 0; i < testrow; i++) {
            if (result.get(0, i) > 0) {
                if (testlabelmat.get(i, 0) == testLabelPositive) {
                    //System.out.println("correct");
                    correct += 1;
                    correctPos += 1;
                } else {
                    //System.out.println("wrong");
                    wrong += 1;
                    wrongPos += 1;
                }
            } else {
                if (testlabelmat.get(i, 0) == testLabelNegative) {
                    //System.out.println("correct");
                    correct += 1;
                    correctNeg += 1;
                } else {
                    //System.out.println("wrong");
                    wrong += 1;
                    wrongNeg += 1;
                }
            }
        }
        System.out.println("Correct is " + correct);
        System.out.println("Wrong is " + wrong);
        double acc = (correct * 100) / testrow;
        System.out.println("Accuracy is " + acc);
        System.out.println("Correct positive is " + correctPos);
        System.out.println("Correct negative is " + correctNeg);
        System.out.println("Wrong positive is " + wrongPos);
        System.out.println("Wrong negative is " + wrongNeg);
    }

    private static SetUpHolder DualAscentSetUp(Matrix mainMatrix){
        //QR Decomposition
        QRHolder qrd = HouseHolderDecomposition2(mainMatrix.getArray());
        Matrix Q = qrd.getQ();
        Matrix R = qrd.getR();
        Matrix betaCapmat = new Matrix(numOfPartialData,1,1.0);
        Matrix Ecap = (Q.transpose()).times(new Matrix(numOfPartialData,1,-1.0));

        Matrix F = new Matrix(numOfPartialData,numOfPartialData);
        Matrix fullRfinalTransposeMat = R.transpose();
        Matrix RgRgTmat = R.times(fullRfinalTransposeMat);
        for (int i = 0; i < nCols; i++) {
            for (int j = 0; j < nCols; j++) {
                F.set(i, j, (RgRgTmat.get(i,j) * (-1)));
            }
        }
        for (int i = 0; i < numOfPartialData; i++) {
            double dval = F.get(i, i);
            dval += (-1.0 / (2.0 * C));
            F.set(i, i, dval);
        }

        SetUpHolder setUpHolder = new SetUpHolder();
        setUpHolder.setQ(Q);
        setUpHolder.setR(R);
        setUpHolder.setBetaCapMat(betaCapmat);
        setUpHolder.seteCap(Ecap);
        setUpHolder.setF(F);
        return setUpHolder;
    }

    public static void main(String[] args) throws IOException {
        numOfPartialData = 2000;
        Scanner fileScanner;
        Matrix[] trainMatrix = new Matrix[numOfClusters];
        Matrix[] testMatrix = new Matrix[numOfClusters];
        Matrix[] testLabelMatrix = new Matrix[numOfClusters];
        SetUpHolder[] setUpHolders = new SetUpHolder[numOfClusters];
        Matrix[] betaCapMats = new Matrix[numOfClusters];

        for(int k=0;k<numOfClusters;k++){
            File trainFile = new File("src/main/resources/a9a/a9a_2000_40_train_meka_"+ String.valueOf(k+1) +".txt");
            trainMatrix[k] = new Matrix(numOfPartialData,nCols);
            fileScanner = new Scanner(trainFile);
            int counter = 0;

            while (fileScanner.hasNextLine()){
                String line = fileScanner.nextLine();
                String[] sarray = line.trim().split(" ");
                for (int i = 0; i < nCols; i++) {
                    trainMatrix[k].set(counter,i,Double.parseDouble(sarray[i]));
                }
                counter++;
            }

            File testFile = new File("src/main/resources/a9a/a9a_1000_40_test_meka_"+ String.valueOf(k+1) +".txt");
            testMatrix[k] = new Matrix(testrow,nCols);
            fileScanner = new Scanner(testFile);
            counter = 0;
            while (fileScanner.hasNextLine()){
                String line = fileScanner.nextLine();
                String[] sarray = line.trim().split(" ");
                for (int i = 0; i < nCols; i++) {
                    testMatrix[k].set(counter,i,Double.parseDouble(sarray[i]));
                }
                counter++;
            }

            File testLabelFile = new File("src/main/resources/a9a/a9a_1000_test_label_"+ String.valueOf(k+1) +".txt");
            testLabelMatrix[k] = new Matrix(testrow,1);
            fileScanner = new Scanner(testLabelFile);
            counter = 0;
            while (fileScanner.hasNextLine()){
                String line = fileScanner.nextLine();
                String[] sarray = line.trim().split(" ");
                testLabelMatrix[k].set(counter,0,Double.parseDouble(sarray[0]));
                counter++;
            }

            setUpHolders[k] = DualAscentSetUp(trainMatrix[k]);
            betaCapMats[k] = setUpHolders[k].getBetaCapMat();
        }

        Matrix prevBetacap = new Matrix(numOfPartialData,1,1.0);

        for (int it = 0; it < maxIteration; it++) {
            AlphaBetaHolder[] holders = new AlphaBetaHolder[numOfClusters];
            Matrix[] betas = new Matrix[numOfClusters];
            for(int k=0;k<numOfClusters;k++){
                holders[k] = AlphaBeta(setUpHolders[k].getF(), betaCapMats[k], setUpHolders[k].geteCap());
                betas[k] = (setUpHolders[k].getQ()).times(holders[k].getBeta());

                for (int i = 0; i < betas[k].getRowDimension(); i++) {
                    for (int j = 0; j < betas[k].getColumnDimension(); j++) {
                        if (betas[k].get(i, j) < 0) {
                            betas[k].set(i, j, 0);
                        }
                    }
                }

                betaCapMats[k] = (setUpHolders[k].getQ().transpose()).times(betas[k]);
            }

            Matrix avgBetaCapMat = betaCapMats[0];
            for(int k=1;k<numOfClusters;k++){
                avgBetaCapMat = avgBetaCapMat.plus(betaCapMats[k]);
            }
            avgBetaCapMat = avgBetaCapMat.times(1.0/numOfClusters);
            Matrix diffbeta = avgBetaCapMat.minus(prevBetacap);
            double error = diffbeta.norm1();
            for (int i = 0; i < numOfPartialData; i++) {
                prevBetacap.set(i, 0, avgBetaCapMat.get(i, 0));
            }

            for(int k=0;k<numOfClusters;k++){
                betaCapMats[k] = avgBetaCapMat;
            }
            System.out.println("############################Here is iteration " + it + "###############");
            System.out.println("this is error " + error);

            if(it%5==0){
                for(int k=0;k<numOfClusters;k++){
                    System.out.println("############### Cluster " + String.valueOf(k+1) + "###############");
                    TestData(holders[k].getAlpha(),(setUpHolders[k].getR()).getMatrix(0, nCols - 1, 0, nCols - 1),testMatrix[k],testLabelMatrix[k]);
                }
            }

            if (error < threshold) {
                break;
            }
        }

//        SparkConf conf = new SparkConf().setMaster("local").setAppName("Word Count");
//        JavaSparkContext sc = new JavaSparkContext(conf);
//        numOfPartialData = totalData / numOfClusters;
//        optStepSize = (floor(1.0/(learnRate*C))-0.5)*learnRate;
//
//        Logger.getLogger("org").setLevel(Level.OFF);
//        Logger.getLogger("akka").setLevel(Level.OFF);
//
//        xAcc = sc.accumulator(new Matrix(totalData, 1), new MatrixAccumulatorParam());
//
//        // Load the training data into a Spark RDD, which is a distributed representation of each line of text
//        //This process does not split data evenly.
//        JavaRDD<String> trainingDataRDD = sc.textFile(trainDataPath, numOfClusters);
//        JavaRDD<Matrix> trainingData = trainingDataRDD.map(new Function<String, Matrix>() {
//                                                                  public Matrix call(String s) {
//                                                                      String[] sarray = s.trim().split(" ");
//                                                                      double[][] values = new double[1][nCols];
//                                                                      for (int i = 0; i < nCols; i++) {
//                                                                          values[0][i] = Double.parseDouble(sarray[i]);
//                                                                      }
//                                                                      return new Matrix(values);
//                                                                  }
//                                                              }
//        ).persist(StorageLevel.MEMORY_ONLY_SER());
//
//        // Load the test data into a Spark RDD, here test data is not multiplied by labels
//        JavaRDD<String> testDataRDD = sc.textFile(testDataPath);
//        JavaRDD<Matrix> testData = testDataRDD.map(new Function<String, Matrix>() {
//                                                         public Matrix call(String s) {
//                                                             String[] singleRecord = s.trim().split(" ");
//                                                             double[][] singleRecordMat = new double[1][nCols];
//                                                             for (int i = 0; i < nCols; i++) {
//                                                                 singleRecordMat[0][i] = Double.parseDouble(singleRecord[i]);
//                                                             }
//                                                             return new Matrix(singleRecordMat);
//                                                         }
//                                                     }
//        ).persist(StorageLevel.MEMORY_ONLY_SER());
//
//        // Load the labels into a Spark RDD
//        JavaRDD<String> testLabelRDD = sc.textFile(testLabelPath);
//        JavaRDD<Double> testLabel = testLabelRDD.map(new Function<String, Double>() {
//                                                           public Double call(String s) {
//                                                               String[] sarray = s.trim().split(" ");
//                                                               return Double.parseDouble(sarray[0]);
//                                                           }
//                                                       }
//        ).persist(StorageLevel.MEMORY_ONLY_SER());
//
//        Matrix testdatamat = new Matrix(testrow, nCols);
//        Matrix testlabelmat = new Matrix(testrow, 1);
//        rowNumberCount = 0;
//        for (Matrix line : testData.take(testrow)) {
//            testdatamat.setMatrix(rowNumberCount, rowNumberCount, 0, nCols - 1, line);
//            rowNumberCount++;
//        }
//        rowNumberCount = 0;
//        for (double line : testLabel.take(testrow)) {
//            testlabelmat.set(rowNumberCount, 0, line);
//            rowNumberCount++;
//        }
//
//        Matrix trainDatamat = new Matrix(totalData,nCols);
//        rowNumberCount = 0;
//        for (Matrix line : trainingData.take(totalData)) {
//            trainDatamat.setMatrix(rowNumberCount, rowNumberCount, 0, nCols - 1, line);
//            rowNumberCount++;
//        }
//
//        //Accumulator to gather parts of R
//        Rgatherac = sc.accumulator(new Matrix(numOfClusters * nCols, nCols), new MatrixAccumulatorParam());
//        //Accumulator to store which cluster have how many records
//        Accumulator<Matrix> recordCountInClusters = sc.accumulator(new Matrix(numOfClusters, 1), new MatrixAccumulatorParam());
//        rowNumberCount = 0;
//        localReflectors = trainingData.mapPartitions(new FlatMapFunction<Iterator<Matrix>, Matrix>() {
//            @Override
//            public Iterator<Matrix> call(Iterator<Matrix> matrixIterator) throws Exception {
//                //If data is not evenly distributed, we are making matrix with slightly larger size.
//                Matrix partitionedMatrix = new Matrix(numOfPartialData + thresholdSpark, nCols);
//                int currentRow = 0;
//                while (matrixIterator.hasNext()) {
//                    Matrix rowMat = matrixIterator.next();
//                    partitionedMatrix.setMatrix(currentRow, currentRow, 0, nCols - 1, rowMat);
//                    currentRow++;
//                }
//                //We are trimming rows which are not used from partitionedMatrix.
//                Matrix trimmedMatrix = partitionedMatrix.getMatrix(0, currentRow - 1, 0, nCols - 1);
//                //Local QR decomposition
//                QRHolder qrd = HouseHolderDecomposition(trimmedMatrix.getArray());
//                //Gathering partial d*d R to RgatherAccumulator
//                Matrix temp = new Matrix(numOfClusters * nCols, nCols);
//                temp.setMatrix(partitionCounter, partitionCounter + nCols - 1, 0, nCols - 1, qrd.getR());
//                partitionCounter += nCols;
//                Rgatherac.add(temp);
//                //Gathering no of instance per cluster posseses
//                Matrix temp2 = new Matrix(numOfClusters, 1);
//                temp2.set(rowNumberCount, 0, currentRow);
//                rowNumberCount += 1;
//                recordCountInClusters.add(temp2);
//                //Returning generated Reflector from each cluster to a RDD
//                List<Matrix> new_List = new ArrayList();
//                new_List.add(qrd.getReflector());
//                return new_List.iterator();
//            }
//        }).persist(StorageLevel.MEMORY_ONLY_SER());
//
//        //Spark uses lazy waiting for mapPartitions method. So we are using this codes to compute QReflectors early
//        JavaRDD<Integer> dummyMap = localReflectors.map(new Function<Matrix, Integer>() {
//            public Integer call(Matrix s) {
//                return 0;
//            }
//        });
//        Integer dummyReduce = dummyMap.reduce(new Function2<Integer, Integer, Integer>() {
//            public Integer call(Integer a, Integer b) {
//                return a+b;
//            }
//        });
//
//        //Generating final R and final R inverse.
//        Matrix RgatherMat = Rgatherac.value();
//        QRHolder qrdholder = HouseHolderDecomposition(RgatherMat.getArray());
//        globalReflector = qrdholder.getReflector();
//        Matrix finalR = qrdholder.getR();
//        ////////////////////////QR Decomposition ends here//////////////////////
//        System.out.println("QR decomposition completes here");
//        //Getting record number in each cluster
//        recordCountInClustersMat = recordCountInClusters.value();
//        //Initializing Beta matrix in 0th iteration
//        Matrix betaCapmat = new Matrix(totalData,1,1.0);
//        betaBroadcast = sc.broadcast(betaCapmat);
//        /*====================Calculating Ecap starts here====================*/
//
//        /*======================Calculating Ecap ends here====================*/
//        Matrix Ecap = Dist_QtX(new Matrix(totalData,1,-1.0));
//        enBroadcast = sc.broadcast(Ecap);
//        List<Matrix> Fs = new ArrayList<>();
//        //Initializing F matrix
//        Matrix[] F = new Matrix[numOfClusters];
//        F[0] = new Matrix((int) recordCountInClustersMat.get(0, 0), (int) recordCountInClustersMat.get(0, 0));
//
//        //Creating RgRgT
//        Matrix fullRfinalTransposeMat = finalR.transpose();
//        Matrix RgRgTmat = finalR.times(fullRfinalTransposeMat);
//        //Building F1
//        //F1 = -((RgRgT)+(1/2C))
//        for (int i = 0; i < nCols; i++) {
//            for (int j = 0; j < nCols; j++) {
//                F[0].set(i, j, (RgRgTmat.get(i,j) * (-1)));
//            }
//        }
//
//        for (int i = 0; i < (int) recordCountInClustersMat.get(0, 0); i++) {
//            double dval = F[0].get(i, i);
//            dval += (-1.0 / (2.0 * C));
//            F[0].set(i, i, dval);
//        }
//        Fs.add(F[0]);
//        //Done creating F1
//
//        //Building F2...P
//        //F2..P = -diag((1/2C))
//        for (int p = 1; p < numOfClusters; p++) {
//            F[p] = new Matrix((int) recordCountInClustersMat.get(p, 0), (int) recordCountInClustersMat.get(p, 0));
//            for (int i = 0; i < (int) recordCountInClustersMat.get(p, 0); i++) {
//                double dval = (-1.0 / (2.0 * C));
//                F[p].set(i, i, dval);
//            }
//            Fs.add(F[p]);
//        }
//        JavaRDD<Matrix> FsRDD = sc.parallelize(Fs, numOfClusters);
//        Matrix prevBetacap = new Matrix(totalData, 1);
//        prevBetacap.setMatrix(0,totalData-1,0,0,betaCapmat);
//        System.out.println("Before iteration");
//        for (int it = 0; it < maxIteration; it++) {
//            alphaList = sc.accumulator(new Matrix(totalData, 1), new MatrixAccumulatorParam());
//            betaList = sc.accumulator(new Matrix(totalData, 1), new MatrixAccumulatorParam());
//            partitionCounter = 0;
//            rowNumberCount = 0;
//            FsRDD.foreachPartition(new VoidFunction<Iterator<Matrix>>() {
//                public void call(Iterator<Matrix> arg0) throws Exception {
//                    Matrix F = arg0.next();
//                    int Flen = F.getRowDimension();
//                    Matrix Beta = betaBroadcast.value().getMatrix(partitionCounter, partitionCounter + Flen - 1, 0, 0);
//                    Matrix E = enBroadcast.value().getMatrix(partitionCounter, partitionCounter + Flen - 1, 0, 0);
//                    AlphaBetaHolder holder = AlphaBeta(F, Beta, E);
//                    Matrix temp = new Matrix(totalData, 1);
//                    Matrix temp2 = new Matrix(totalData, 1);
//                    temp.setMatrix(partitionCounter, partitionCounter + Flen - 1, 0, 0, holder.getAlpha());
//                    alphaList.add(temp);
//                    temp2.setMatrix(rowNumberCount, rowNumberCount + Flen - 1, 0, 0, holder.getBeta());
//                    betaList.add(temp2);
//                    partitionCounter += Flen;
//                    rowNumberCount += Flen;
//                }
//            });
//            Matrix Betamat = Dist_QX(betaList.value());
//            for (int i = 0; i < Betamat.getRowDimension(); i++) {
//                for (int j = 0; j < Betamat.getColumnDimension(); j++) {
//                    if (Betamat.get(i, j) < 0) {
//                        Betamat.set(i, j, 0);
//                    }
//                }
//            }
//            betaCapmat = Dist_QtX(Betamat);
//            Matrix diffbeta = betaCapmat.minus(prevBetacap);
//            double error = diffbeta.norm1();
//            for (int i = 0; i < totalData; i++) {
//                prevBetacap.set(i, 0, betaCapmat.get(i, 0));
//            }
//            System.out.println("############################Here is iteration " + it + "###############");
//            System.out.println("this is error " + error);
//            betaBroadcast = sc.broadcast(betaCapmat);
//            if(it%5==0){
//                TestData(alphaList.value(),finalR,testdatamat,testlabelmat);
//            }
//            if (error < threshold) {
//                break;
//            }
//        }
//        Matrix alphatotalmat = alphaList.value();
//        TestData(alphatotalmat,finalR,testdatamat,testlabelmat);
//        //Matrix support_vectors = Dist_QX(alphatotalmat);
//        Matrix weightmat = (finalR.transpose()).times(alphatotalmat.getMatrix(0, nCols - 1, 0, 0));
//        Matrix transposeweightmat = weightmat.transpose();
//        Matrix result = transposeweightmat.times(trainDatamat.transpose());
//        for(int i=0;i<result.getColumnDimension();i++){
//            //if(result.get(i,0)<=1 && result.get(i,0)>=-1){
//            System.out.println(i+" "+result.get(0,i));
//            //}
//        }
//
//
//
//            /*final XYSeriesDemo demo = new XYSeriesDemo("Beta change vs Iteration", new Matrix(betadifara), "Amount of beta change");
//            demo.pack();
//            RefineryUtilities.centerFrameOnScreen(demo);
//            demo.setVisible(true);*/


    }

}
