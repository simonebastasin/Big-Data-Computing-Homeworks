import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public class G098HW2 {

	// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
	// Input reading methods
	// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

	public static Vector strToVector(String str) {
		String[] tokens = str.split(",");
		double[] data = new double[tokens.length];
		for (int i = 0; i < tokens.length; i++) {
			data[i] = Double.parseDouble(tokens[i]);
		}
		return Vectors.dense(data);
	}

	public static ArrayList<Vector> readVectorsSeq(String filename) throws IOException {
		if (Files.isDirectory(Paths.get(filename))) {
			throw new IllegalArgumentException("readVectorsSeq is meant to read a single file.");
		}    
		ArrayList<Vector> result = new ArrayList<>();
		Files.lines(Paths.get(filename))
			.map(str -> strToVector(str))
			.forEach(e -> result.add(e));
		return result;
	}
	
	
	// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
	// kcenter methods
	// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

	public static ArrayList<Vector> SeqWeightedOutliers(ArrayList<Vector> P, ArrayList<Long> W, int k, int z, int alpha) {
		double r, temp;
        ArrayList<Vector> uncovered, S;
        long wz, max, ballWeight;
        int nGuesses = 0, newCenter = -1, index;
		
		// Calculation of r_min (initial guess of r)
		r = Math.sqrt(Vectors.sqdist(P.get(0), P.get(1)));
		for (int i = 0; i < k + z; i++) {
			for (int j = i + 1; j < k + z + 1; j++) {
				temp = Math.sqrt(Vectors.sqdist(P.get(i), P.get(j)));
				if (temp < r) {
					r = temp;
				}
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
            while(S.size() < k && wz > 0) {
                max = 0;
                for (int i = 0; i < P.size(); i++) {
					ballWeight = 0;
                    for (int j = 0; j < uncovered.size(); j++) {
                        if (Math.sqrt(Vectors.sqdist(P.get(i), uncovered.get(j))) <= (1 + 2 * alpha) * r) {
							ballWeight += W.get(j);
						}
                    }
                    if (ballWeight > max) {
                        max = ballWeight;
                        newCenter = i;
                    }
                }
                S.add(P.get(newCenter));
				index = 0;
				for (int j = 0; index < uncovered.size(); j++) {
					if (Math.sqrt(Vectors.sqdist(P.get(newCenter), uncovered.get(index))) <= (3 + 4 * alpha) * r) {
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
	
	public static double ComputeObjective(ArrayList<Vector> P, ArrayList<Vector> S, int z) {
        double r, temp;
		ArrayList<Double> distances = new ArrayList<Double>();

        // Calculation of distances between points and their own centers
        for (int i = 0; i < P.size(); i++) {
            r = Math.sqrt(Vectors.sqdist(P.get(i), S.get(0)));
            for (int j = 1; j < S.size(); j++) {
                temp = Math.sqrt(Vectors.sqdist(P.get(i), S.get(j)));
                if (temp < r) {
                    r = temp;
				}
            }
            distances.add(r);
        }
		
		// Return point with biggest distance (not considering outliers)
		Collections.sort(distances);
        return distances.get(distances.size() - z - 1);
    }
	
	
	// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
	// Main program
	// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
	
    public static void main(String[] args) throws IOException {

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // CHECKING NUMBER OF CMD LINE PARAMETERS
        // Parameters are: <path_to_file>, k, z
        // k = num_centers (Integer)
        // z = num_allowed_outliers (Integer)
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        if (args.length != 3) {
            throw new IllegalArgumentException("USAGE: file_path num_centers num_allowed_outliers");
        }

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // INPUT READING
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        // Read points in input file (into an ArrayList<Vector>)
        ArrayList<Vector> inputPoints = readVectorsSeq(args[0]);
        // Read number of centers
        int k = Integer.parseInt(args[1]);
        // Read number of allowed outliers
        int z = Integer.parseInt(args[2]);

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SETTING GLOBAL VARIABLES
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

		ArrayList<Long> weights = new ArrayList<Long>();
		for (int i = 0; i < inputPoints.size(); i++) {
			weights.add((long) 1);
		}

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // TASK 1
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
		
		System.out.println("Input size n = " + inputPoints.size());
        System.out.println("Number of centers k = " + k);
        System.out.println("Number of outliers z = " + z);
        long time = System.currentTimeMillis();
		ArrayList<Vector> solution = SeqWeightedOutliers(inputPoints, weights, k, z, 0);
        time = System.currentTimeMillis() - time;

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // TASK 2
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
	
		double objective = ComputeObjective(inputPoints, solution, z);
        System.out.println("Objective function = " + objective);
        System.out.println("Time of SeqWeightedOutliers = " + time);
	}
}