import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.io.IOException;
import java.util.*;

public class G098HW1{

    public static void main(String[] args) throws IOException {

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // CHECKING NUMBER OF CMD LINE PARAMETERS
        // Parameters are: K, H, S, <path_to_file>
        // K = num_partitions (Integer)
        // H = num_products (Integer)
        // S = country_name (String)
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        if (args.length != 4) {
            throw new IllegalArgumentException("USAGE: num_partitions num_products country_name file_path");
        }

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SPARK SETUP
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        SparkConf conf = new SparkConf(true).setAppName("G098HW1");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // INPUT READING
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        // Read number of partitions
        int K = Integer.parseInt(args[0]);
        // Read number of products
        int H = Integer.parseInt(args[1]);
        // Read country name
        String S = args[2];
        // Read input file and subdivide it into K random partitions
        JavaRDD<String> rawData = sc.textFile(args[3]).repartition(K).cache();

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SETTING GLOBAL VARIABLES
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        long numrows, numpairs;
        JavaPairRDD<String, Integer> productCustomer, productPopularity1, productPopularity2;
        Random randomGenerator = new Random();

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // TASK 1
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        numrows = rawData.count();
        System.out.println("Number of rows = " + numrows);

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // TASK 2
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        productCustomer = rawData
                .flatMapToPair((row) -> {    // <-- MAP PHASE (R1)
                    String[] tokens = row.split(",");
                    ArrayList<Tuple2<String, Integer>> pairs = new ArrayList<>();
                    if (Integer.parseInt(tokens[3]) > 0) { // check Quantity > 0
                        if (S.equals("all") || tokens[7].equals(S)) { // check Country
                            pairs.add(new Tuple2<>(tokens[1]+"&"+tokens[6], Integer.parseInt(tokens[6])));
                        }
                    }
                    return pairs.iterator();
                })
                .groupByKey()     // <-- SHUFFLE+GROUPING
                .flatMapToPair((element) -> { // <-- REDUCE PHASE (R1)
                    ArrayList<Tuple2<String, Integer>> pairs = new ArrayList<>();
                    String tmp[] = element._1().split("&");
                    pairs.add(new Tuple2<>(tmp[0], element._2().iterator().next()));
                    return pairs.iterator();
                });
        numpairs = productCustomer.count();
        System.out.println("Product-Customer Pairs = " + numpairs);

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // TASK 3
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        productPopularity1 = productCustomer
                .mapPartitionsToPair((element) -> {    // <-- MAP PHASE (R1)
                    HashMap<String, Integer> counts = new HashMap<>();
                    ArrayList<Tuple2<String, Integer>> pairs = new ArrayList<>();
                    while (element.hasNext()){
                        Tuple2<String, Integer> tuple = element.next();
                        counts.put(tuple._1(), 1 + counts.getOrDefault(tuple._1(), 0));
                    }
                    for (Map.Entry<String, Integer> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                .groupByKey()     // <-- SHUFFLE+GROUPING
                .mapValues((it) -> { // <-- REDUCE PHASE (R1)
                    int sum = 0;
                    for (int c : it) {
                        sum += c;
                    }
                    return sum;
                });

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // TASK 4
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        productPopularity2 = productCustomer
                .mapToPair((element) -> new Tuple2<String, Integer>(element._1(), 1))    // <-- MAP PHASE (R1)
                .reduceByKey((x, y) -> x+y);    // <-- REDUCE PHASE (R1)

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // TASK 5
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        if (H > 0) {
            JavaPairRDD<Integer, String> productPopularity3;
            productPopularity3 = productPopularity1
                    .mapToPair((element) -> new Tuple2<Integer, String>(element._2(), element._1()));    // <-- MAP PHASE (R1)
            List<Tuple2<Integer, String>> popularityList = productPopularity3.sortByKey(false).take(H);
            System.out.println("Top " + H + " Products and their Popularities");
            for (Tuple2<Integer, String> element : popularityList) {
                System.out.print("Product " + element._2() + " Popularity " + element._1() + "; ");
            }
        }

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // TASK 6
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        else if (H == 0) {
            List<Tuple2<String, Integer>> fullPopularityList_1 = productPopularity1.sortByKey(true).collect();
            System.out.println("productPopularity1: ");
            for (Tuple2<String, Integer> element : fullPopularityList_1) {
                System.out.print("Product: " + element._1() + " Popularity: " + element._2() + "; ");
            }
            List<Tuple2<String, Integer>> fullPopularityList_2 = productPopularity2.sortByKey(true).collect();
            System.out.println("\nproductPopularity2: ");
            for (Tuple2<String, Integer> element : fullPopularityList_2) {
                System.out.print("Product: " + element._1() + " Popularity: " + element._2() + "; ");
            }
        }
    }
}