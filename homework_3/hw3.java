import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.util.*;

public class G098HW3 {

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // MAIN PROGRAM
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static void main(String[] args) throws Exception {

        if (args.length != 4) {
            throw new IllegalArgumentException("USAGE: filepath k z L");
        }

        // ----- Initialize variables
        String filename = args[0];
        int k = Integer.parseInt(args[1]);
        int z = Integer.parseInt(args[2]);
        int L = Integer.parseInt(args[3]);
        long start, end; // variables for time measurements

        // ----- Set Spark Configuration
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);
        SparkConf conf = new SparkConf(true).setAppName("MR k-center with outliers");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // ----- Read points from file
        start = System.currentTimeMillis();
        JavaRDD<Vector> inputPoints = sc.textFile(args[0], L)
                .map(x-> strToVector(x))
                .repartition(L)
                .cache();
        long N = inputPoints.count();
        end = System.currentTimeMillis();

        // ----- Print input parameters
        System.out.println("File : " + filename);
        System.out.println("Number of points N = " + N);
        System.out.println("Number of centers k = " + k);
        System.out.println("Number of outliers z = " + z);
        System.out.println("Number of partitions L = " + L);
        System.out.println("Time to read from file: " + (end-start) + " ms");

        // ---- Solve the problem
        ArrayList<Vector> solution = MR_kCenterOutliers(inputPoints, k, z, L);

        // ---- Compute the value of the objective function
        start = System.currentTimeMillis();
        double objective = computeObjective(inputPoints, solution, z);
        end = System.currentTimeMillis();
        System.out.println("Objective function = " + objective);
        System.out.println("Time to compute objective function: " + (end-start) + " ms");
    }

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // AUXILIARY METHODS
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Method strToVector: input reading
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Method euclidean: distance function
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static double euclidean(Vector a, Vector b) {
        return Math.sqrt(Vectors.sqdist(a, b));
    }

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Method MR_kCenterOutliers: MR algorithm for k-center with outliers
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Vector> MR_kCenterOutliers (JavaRDD<Vector> points, int k, int z, int L) {
        // Measure and print times taken by Round 1 and Round 2, separately
        long start, middle, end;
        start = System.currentTimeMillis(); // time 1/3

        //------------- ROUND 1 ---------------------------

        JavaRDD<Tuple2<Vector,Long>> coreset = points.mapPartitions(x -> {
            ArrayList<Vector> partition = new ArrayList<>();
            while (x.hasNext()) partition.add(x.next());
            ArrayList<Vector> centers = kCenterFFT(partition, k+z+1);
            ArrayList<Long> weights = computeWeights(partition, centers);
            ArrayList<Tuple2<Vector,Long>> c_w = new ArrayList<>();
            for(int i = 0; i < centers.size(); ++i) {
                Tuple2<Vector, Long> entry = new Tuple2<>(centers.get(i), weights.get(i));
                c_w.add(i,entry);
            }
            return c_w.iterator();
        }); // END OF ROUND 1

        //------------- ROUND 2 ---------------------------

        ArrayList<Tuple2<Vector, Long>> elems = new ArrayList<>((k + z) * L);
        elems.addAll(coreset.collect());
        middle = System.currentTimeMillis(); // time 2/3
        ArrayList<Vector> P = new ArrayList<>();
        ArrayList<Long> W = new ArrayList<>();
        for (Tuple2<Vector, Long> e : elems) {
            P.add(e._1());
            W.add(e._2());
        }
        // Compute the final solution (run SeqWeightedOutliers with alpha=2)
        ArrayList<Vector> solution = SeqWeightedOutliers(P, W, k, z, 2);
        // END OF ROUND 2

        end = System.currentTimeMillis(); // time 3/3
        System.out.println("Time Round 1: " + (middle-start) + " ms");
        System.out.println("Time Round 2: " + (end-middle) + " ms");
        // Return the final solution
        return solution;
    }

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Method kCenterFFT: Farthest-First Traversal
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Vector> kCenterFFT (ArrayList<Vector> points, int k) {
        final int n = points.size();
        double[] minDistances = new double[n];
        Arrays.fill(minDistances, Double.POSITIVE_INFINITY);
        ArrayList<Vector> centers = new ArrayList<>(k);
        Vector lastCenter = points.get(0);
        centers.add(lastCenter);
        double radius = 0;

        for (int iter = 1; iter < k; iter++) {
            int maxIdx = 0;
            double maxDist = 0;
            for (int i = 0; i < n; i++) {
                double d = euclidean(points.get(i), lastCenter);
                if (d < minDistances[i]) {
                    minDistances[i] = d;
                }
                if (minDistances[i] > maxDist) {
                    maxDist = minDistances[i];
                    maxIdx = i;
                }
            }
            lastCenter = points.get(maxIdx);
            centers.add(lastCenter);
        }
        return centers;
    }

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Method computeWeights: compute weights of coreset points
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Long> computeWeights(ArrayList<Vector> points, ArrayList<Vector> centers) {
        Long weights[] = new Long[centers.size()];
        Arrays.fill(weights, 0L);
        for(int i = 0; i < points.size(); ++i) {
            double tmp = euclidean(points.get(i), centers.get(0));
            int mycenter = 0;
            for(int j = 1; j < centers.size(); ++j) {
                if(euclidean(points.get(i),centers.get(j)) < tmp) {
                    mycenter = j;
                    tmp = euclidean(points.get(i), centers.get(j));
                }
            }
            // System.out.println("Point = " + points.get(i) + " Center = " + centers.get(mycenter));
            weights[mycenter] += 1L;
        }
        ArrayList<Long> fin_weights = new ArrayList<>(Arrays.asList(weights));
        return fin_weights;
    }

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Method SeqWeightedOutliers: sequential k-center with outliers
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Vector> SeqWeightedOutliers(ArrayList<Vector> P, ArrayList<Long> W, int k, int z, int alpha) {
        double r, temp;
        ArrayList<Vector> uncovered, S;
        long wz, max, ballWeight;
        int nGuesses = 0, newCenter = -1, index;
        Vector tempVec;

        // Calculation of r_min (initial guess of r)
        r = Math.sqrt(Vectors.sqdist(P.get(0), P.get(1)));
        for (int i = 0; i < k + z; i++) {
            tempVec = P.get(i);
            for (int j = i + 1; j < k + z + 1; j++) {
                temp = Math.sqrt(Vectors.sqdist(tempVec, P.get(j)));
                if (temp < r) r = temp;
            }
        }
        r /= 2;
        System.out.println("Initial guess = " + r);

        // Algorithm kcenterOUT (guesses of r)
        while (true) {
            nGuesses++;
            uncovered = (ArrayList<Vector>) P.clone();
            S = new ArrayList<Vector>();
            wz = 0;
            for (long i : W) {
				wz += i;
			}
            while (S.size() < k && wz > 0) {
                max = 0;
                for (int i = 0; i < P.size(); i++) {
                    tempVec = P.get(i);
                    ballWeight = 0;
                    for (int j = 0; j < uncovered.size(); j++) {
                        if (Math.sqrt(Vectors.sqdist(tempVec, uncovered.get(j))) <= ((2 * alpha + 1 ) * r)) {
                            ballWeight += W.get(j);
                        }
                    }
                    if (ballWeight > max) {
                        max = ballWeight;
                        newCenter = i;
                    }
                }
                tempVec = P.get(newCenter);
                S.add(tempVec);
                index = 0;
                for (int j = 0; index < uncovered.size(); j++) {
                    if (Math.sqrt(Vectors.sqdist(tempVec, uncovered.get(index))) <= ((4 * alpha +3) * r)) {
                        uncovered.remove(index);
                        wz -= W.get(j);
                    }
                    else {
                        index++;
                    }
                }
            }
            if (wz <= z) {
                System.out.println("Final guess = " + r);
                System.out.println("Number of guesses = " + nGuesses);
                return S;
            }
            else {
                r *= 2;
            }
        }
    }

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Method computeObjective: computes objective function
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static double computeObjective (JavaRDD<Vector> points, ArrayList<Vector> centers, int z) {
        // RDD that at the end contains pairs (distance, 1) of the
        // L*(z+1) higher distances of each point from its center
        JavaPairRDD<Double, Integer> distancePairs = points.mapPartitionsToPair(x -> { // <-- MAP PHASE (R1)
            // Array of distances of each point from its center
            ArrayList<Double> distances = new ArrayList<Double>();
            // Array of z+1 pairs (0, distance) to return
            ArrayList<Tuple2<Integer, Double>> pairs = new ArrayList<>();
            // For each point
            while (x.hasNext()){
                Vector current = x.next(); // current point
                // find distance of current point from its center
                double r = euclidean(current, centers.get(0));
                for (Vector center : centers) {
                    if (euclidean(current, center) < r) {
                        r = euclidean(current, center);
                    }
                }
                // add distance found to array
                distances.add(r);
            }
            Collections.sort(distances); // sort distances in decreasing order
            // For each of the z+1 higher distances found
            for (int i = 0; i < z + 1; i++){
                Double tempDist = distances.get(distances.size() - i - 1);
                // add pair (0, distance)
                pairs.add(new Tuple2<>(0, tempDist));
            }
            return pairs.iterator();
        })
                .groupByKey() // <-- SHUFFLE+GROUPING
                .flatMapToPair((element) -> { // <-- REDUCE PHASE (R1)
                    // Array of maximum L*(z+1) pairs (distance, 1) to return
                    ArrayList<Tuple2<Double, Integer>> pairs = new ArrayList<>();
                    // For each distance
                    for (Double dist : element._2()) {
						// add pair (distance, 1)
                        pairs.add(new Tuple2<>(dist, 1));
                    }
                    return pairs.iterator();
                })
                .sortByKey(false); // <-- SORTING (decreasing order)
				
        // Take the general z+1 higher distances found
        List<Tuple2<Double, Integer>> c = distancePairs.take(z + 1);

        // Return the highest distance without considering outliers
        return c.get(c.size() - 1)._1();
    }
}