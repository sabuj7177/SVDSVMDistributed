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
public class QRHolder {
    Matrix Q;
    Matrix R;
    Matrix reflector;

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

    public Matrix getReflector() {
        return reflector;
    }

    public void setReflector(Matrix reflector) {
        this.reflector = reflector;
    }    
}
