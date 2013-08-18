package com.cts;

/**
 * Created with IntelliJ IDEA.
 * User: Colin Stearns
 * Date: 5/1/13
 * Time: 11:12 AM
 * To change this template use File | Settings | File Templates.
 */
public class Analysis {
    public static double averageDistance(double[][] points, int pointIndex){
        double distance = 0;
        for (int i = 0; i < points.length; i++){
            distance += getDistance(points[i],points[pointIndex]);
        }
        return distance / (points.length - 1);// -1 so as to not count distance to self, which is 0.
    }

    public static double getDistance (double[] a, double[] b){
        double squaredDist = 0;
        if (a.length != b.length){
            System.out.println("Cannot take distances between points of different dimension");
            return -1;
        }
        for (int i = 0; i < a.length; i++){
            squaredDist += Math.pow(a[i] - b[i],2);
        }
        return Math.sqrt(squaredDist);
    }
}
