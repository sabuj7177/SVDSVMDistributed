import Jama.Matrix;
import org.apache.spark.AccumulatorParam;

/**
 * This class supports Accumulator of type String. It implements an element-by-element add operation for
 * two String
 * 
 * @author Tarek Elgamal
 *
 */

public class MatrixSumParam implements AccumulatorParam<Matrix> {

	public Matrix addInPlace(Matrix arg0, Matrix arg1) {
		return arg0.plus(arg1);
	}

	public Matrix zero(Matrix arg0) {
		return arg0;
	}

	public Matrix addAccumulator(Matrix arg0, Matrix arg1) {
		return addInPlace(arg0,arg1);
	}
	

}
