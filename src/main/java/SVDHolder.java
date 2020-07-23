import Jama.Matrix;

public class SVDHolder {
    Matrix U;
    Matrix S;

    public Matrix getU() {
        return U;
    }

    public void setU(Matrix u) {
        U = u;
    }

    public Matrix getS() {
        return S;
    }

    public void setS(Matrix s) {
        S = s;
    }
}
