package com.cts;

import java.util.Random;
/*
4/20/2013
com.cts.RbmLayer consists of a layer of initialized neurons which do not speak to each other, but instead
receive inputs from the above layer, multiply each input by a weight, and sum the results. The
output of the neurons is binary, with probability of firing given as a bias plus  the logistic
function (1/(1+Exp[-x])) where x is the weighted sum over the neuron's inputs.

The layer is implemented as a n x m matrix of (double) weights, with n the number of neurons, m
the number of inputs (i.e. neurons in the layer above). Additionally, a size-n array of double-
 valued biases are stored for the layer. If the layer is the first layer in a net, it will
 simply take on probability equal to the the value of the inputs (which should be between 0 and
 1).
*/

public class RbmLayer {
    private double[] biases;
    private double[] state;
    private double[] probabilities;
    private int layerSize;
    private Connection upper;
    private Connection lower;
    private static int gibbsSteps = 10;
    private static double biasTrainingRate = 0.1;

    public RbmLayer(int neuronCount){
        biases = new double[neuronCount];
        state = new double[neuronCount];
        layerSize = neuronCount;
        for (int i = 0; i < neuronCount; i++){
            state[i] = 0;
            biases[i] = 0;
        }
    }

    public void connectUp (RbmLayer upperLayer){
        upper = new Connection(upperLayer, this);
        upperLayer.connectDown(upper);
    }

    public void connectUp(Connection upper){
        this.upper = upper;
    }

    public void connectDown (RbmLayer lowerLayer){
        lower = new Connection(this, lowerLayer);
        lowerLayer.connectUp(lower);
    }

    public void connectDown (Connection lower){
        this.lower = lower;
    }

    public void updateUp(){
        double[] sums = lower.getSumsUpper();
        probabilities = getProbabilities(sums);
        state = getOutputs(probabilities);
    }

    public void updateDown(){
        double[] sums = upper.getSumsLower();
        probabilities = getProbabilities(sums);
        state = getOutputs(probabilities);
    }

    public double[] getOutputs (double[] probabilities){
        double[] thresholds = getThresholds(probabilities.length);
        double[] outputs = new double[layerSize];
        for (int i = 0; i < outputs.length; i++){
            outputs[i] = (probabilities[i] > thresholds[i]) ? 1.0 : 0.0;
        }
        return outputs;
    }

    public double[] getState(){
        return state;
    }

    public void initialize(double[] probabilities){
        this.probabilities = probabilities;
        this.state = getOutputs(probabilities);
    }

    public void setState(double[] state){
        this.state = state;
    }

    public double[] getProbabilities () {
        return probabilities;
    }

    public double[] getProbabilities (double[] sums){
        double[] probabilities = new double[layerSize];
        for (int i = 0; i < layerSize; i++){
            probabilities[i] = 1.0/(1.0 + Math.exp(-1 * sums[i]));
        }
        return probabilities;
    }

    //for use as the first layer in the net
    public void setProbabilities (double[] probabilities){
        this.probabilities = probabilities;
    }

    private double[] getThresholds (int n){
        Random rand = new Random();
        double[] thresholds = new double[n];
        for (int i = 0; i < n; i++){
            thresholds[i] = rand.nextDouble();
        }
        return thresholds;
    }

    public int getLayerSize () {
        return layerSize;
    }

    public double[] getBiases () {
        return biases;
    }

    public Connection[] getConnections (){
        Connection[] connections = {upper,lower};
        return connections;
    }

    private static double[] copy (double[] a){
        double[] copy = new double[a.length];
        for (int i = 0; i < a.length; i++){
            copy[i] = a[i];
        }
        return copy;
    }

    public double[] train(RbmLayer trainer) {
        if (upper != trainer.getConnections()[1]){//if this upper isn't trainer's lower
               System.out.println("Failure: No connection between trainer and trainee.");
               return null;
        } else {
            double[] trainerDataState = copy(trainer.getState());
            double[] trainerDataProbabilities = copy(trainer.getProbabilities());
            updateDown();
            double[] traineeDataProbabilities = copy(probabilities);
            double[] trainerModelProbabilities;
            double[] traineeModelProbabilities;
            for (int i = 0; i < gibbsSteps; i++){
                trainer.updateUp();
                updateDown();
            }
            trainerModelProbabilities = copy(trainer.getProbabilities());
            traineeModelProbabilities = copy(probabilities);
            upper.updateWeights(trainerDataProbabilities, traineeDataProbabilities,
                    trainerModelProbabilities, traineeModelProbabilities);
        //    double biasChange = (-1)*biases[0];
            for (int i = 0; i < biases.length; i++){
                biases[i] += biasTrainingRate * (traineeDataProbabilities[i]
                        - traineeModelProbabilities[i]);
            }
        //    biasChange += biases[0];
            trainer.setState(trainerDataState);
            trainer.setProbabilities(trainerDataProbabilities);

        //    System.out.println("bias 0 change: " + biasChange);
        //   System.out.println("distance: " + Analysis.getDistance(trainerDataProbabilities,traineeModelProbabilities));
            return trainerModelProbabilities;
        }
    }
}