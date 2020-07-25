import Jama.Matrix;

import java.io.Serializable;

public class PerClusterInfo implements Serializable {
    private Matrix localReflector;
    private Matrix F;
    private Matrix e;
    private Matrix beta;
    private Matrix prevBeta;
    private Matrix alphaCap;
    private int dataCount;
    private int clusterNumber;

    public Matrix getLocalReflector() {
        return localReflector;
    }

    public void setLocalReflector(Matrix localReflector) {
        this.localReflector = localReflector;
    }

    public Matrix getF() {
        return F;
    }

    public void setF(Matrix f) {
        F = f;
    }

    public Matrix getE() {
        return e;
    }

    public void setE(Matrix e) {
        this.e = e;
    }

    public Matrix getBeta() {
        return beta;
    }

    public void setBeta(Matrix beta) {
        this.beta = beta;
    }

    public Matrix getPrevBeta() {
        return prevBeta;
    }

    public void setPrevBeta(Matrix prevBeta) {
        this.prevBeta = prevBeta;
    }

    public Matrix getAlphaCap() {
        return alphaCap;
    }

    public void setAlphaCap(Matrix alphaCap) {
        this.alphaCap = alphaCap;
    }

    public int getDataCount() {
        return dataCount;
    }

    public void setDataCount(int dataCount) {
        this.dataCount = dataCount;
    }

    public int getClusterNumber() {
        return clusterNumber;
    }

    public void setClusterNumber(int clusterNumber) {
        this.clusterNumber = clusterNumber;
    }

    public void setPartialE(int startRow, int endRow, int startCol, int endCol, Matrix partialE){
        this.e.setMatrix(startRow, endRow, startCol, endCol, partialE);
    }

    public void setPartialBeta(int startRow, int endRow, int startCol, int endCol, Matrix partialBeta){
        this.beta.setMatrix(startRow, endRow, startCol, endCol, partialBeta);
    }

    public void copyBetaToPrevBeta(Matrix beta){
        for(int i=0;i<this.dataCount;i++){
            this.prevBeta.set(i,0,beta.get(i,0));
        }
    }
}
