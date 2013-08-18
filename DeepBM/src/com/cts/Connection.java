package com.cts;

import java.util.Random;

/**
 * Created with IntelliJ IDEA.
 * User: Colin Stearns
 * Date: 4/20/13
 * Time: 6:49 PM
 * To change this template use File | Settings | File Templates.
 */
public class Connection {
    private double[][] weights;
    private RbmLayer upper;
    private RbmLayer lower;
    private static double trainingRate = 0.1;
    private static double initWeightSD = 0.01;//initial weight standard distribution

    public Connection (RbmLayer upper, RbmLayer lower){
        this.upper = upper;
        this.lower = lower;
        Random rand = new Random();
        weights = new double[upper.getLayerSize()][lower.getLayerSize()];
        for (int i = 0; i < upper.getLayerSize(); i++){
            for (int j = 0; j < lower.getLayerSize(); j++){
                weights[i][j] = initWeightSD * rand.nextGaussian();
            }
        }
    }

    public Connection (RbmLayer upper, RbmLayer lower, double[][] weights){
        this.upper = upper;
        this.lower = lower;
        this.weights = weights;
    }

    public double[] getSumsUpper (){
        double[] sums = new double[upper.getLayerSize()];
        double[] state = lower.getState();
        for (int i = 0; i < upper.getLayerSize(); i++){
            for (int j = 0; j < lower.getLayerSize(); j++){
                sums[i] += state[j] * weights[i][j];
            }
        }
        return sums;
    }

    public double[] getSumsLower () {
        double[] sums = new double[lower.getLayerSize()];
        double[] state = upper.getState();
        for (int i = 0; i < upper.getLayerSize(); i++){
            for (int j = 0; j < lower.getLayerSize(); j++){
                sums[j] += state[i] * weights[i][j];
            }
        }
        return sums;
    }

    public void updateWeights(double[] upperDataProbs, double[] lowerDataProbs,
                              double[] upperModelProbs, double[] lowerModelProbs) {
        double zerozeroChange = -1*weights[0][0];
        for (int i = 0; i < upper.getLayerSize(); i++){
            for (int j = 0; j < lower.getLayerSize(); j++){
                weights[i][j] += trainingRate * (upperDataProbs[i]*lowerDataProbs[j] -
                              upperModelProbs[i]*lowerModelProbs[j]);
            }
        }
        zerozeroChange += weights[0][0];
/*
        System.out.println("Data V 0: " + upperDataProbs[0]);
        System.out.println("Data H 0: " + lowerDataProbs[0]);
        System.out.println("Data product: " + upperDataProbs[0]*lowerDataProbs[0]);
        System.out.println("Model V 0: " + upperModelProbs[0]);
        System.out.println("Model H 0: " + lowerModelProbs[0]);
        System.out.println("Model product: " + upperModelProbs[0]*lowerModelProbs[0]);
        System.out.println("Weight: " + weights[0][0]);
        System.out.println("Weight change: " + zerozeroChange);
*/
    }
}
