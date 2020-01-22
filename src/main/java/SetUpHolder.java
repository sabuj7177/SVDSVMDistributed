/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

import Jama.Matrix;

/**
 *
 * @author Sabuj
 */
public class SetUpHolder {
    private Matrix Q;
    private Matrix R;
    private Matrix betaCapMat;
    private Matrix eCap;
    private Matrix F;

    public Matrix getQ() {
        return Q;
    }

    public void setQ(Matrix Q) {
        this.Q = Q;
    }

    public Matrix getR() {
        return R;
    }

    public void setR(Matrix R) {
        this.R = R;
    }

    public Matrix getBetaCapMat() {
        return betaCapMat;
    }

    public void setBetaCapMat(Matrix betaCapMat) {
        this.betaCapMat = betaCapMat;
    }

    public Matrix geteCap() {
        return eCap;
    }

    public void seteCap(Matrix eCap) {
        this.eCap = eCap;
    }

    public Matrix getF() {
        return F;
    }

    public void setF(Matrix f) {
        F = f;
    }
}
