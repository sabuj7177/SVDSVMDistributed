import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

public class SingleSparkContext {
    private static SparkConf conf = new SparkConf().setMaster("local").setAppName("Word Count");
    public static JavaSparkContext sc = new JavaSparkContext(conf);
}
