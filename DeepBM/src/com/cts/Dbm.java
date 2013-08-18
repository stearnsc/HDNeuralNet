package com.cts;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Dbm {
    private List<RbmLayer> layers;
    private int inputSize;
    private static int trainingRounds = 100000;

    public Dbm (int[] layerSizes){
        inputSize = layerSizes[0];
        layers = new ArrayList<RbmLayer>(layerSizes.length);
        layers.add(new RbmLayer(layerSizes[0]));
        RbmLayer newLayer;
        for (int i = 1; i < layerSizes.length; i++){
            newLayer = new RbmLayer(layerSizes[i]);
            newLayer.connectUp(layers.get(i - 1));
            layers.add(newLayer);
        }
    }

    public List<RbmLayer> getLayers (){
        return layers;
    }

    //returns neuron firing probability (propogated through layers) for each layer
    public List<double[]> getOutputData (double[] input){
        if (input.length != inputSize){
            System.out.println("invalid input size");
            return null;
        } else {
            List<double[]> outputData = new ArrayList<double[]>();
            RbmLayer inputLayer = layers.get(0);
            inputLayer.setProbabilities(input);
            inputLayer.setState(inputLayer.getOutputs(input));
            RbmLayer thisLayer;
            for (int i = 1; i < layers.size(); i++){
                thisLayer = layers.get(i);
                thisLayer.updateDown();
                outputData.add(thisLayer.getProbabilities());
            }
            return outputData;
        }
    }

    public double[] getOutput (double[] input){
        List<double[]> output = getOutputData(input);
        return output.get(output.size() - 1);
    }

    public double[] getModel (double[] input){
        double[] output = getOutput(input);
        layers.get(layers.size() - 1).setProbabilities(output);
        for (int i = layers.size() - 2; i >= 0; i--){
            layers.get(i).updateUp();
        }
        return layers.get(0).getProbabilities();
    }

    public void train (double[][] input){
        double[] model;
        double distance;
        for (int i = 1; i < layers.size(); i++){
            for (int j = 0; j < trainingRounds; j++){
                distance = 0;
                for (int k = 0; k < input.length; k++){
                    layers.get(0).initialize(input[k]);
                    for (int l = 2; l < i; l++){//the trainer for 1 is data-driven, so start at 2
                        layers.get(l).updateDown();
                    }
                    model = layers.get(i).train(layers.get(i - 1));
                    distance += Analysis.getDistance(model, input[k]);
                }
                System.out.println("Average Distance: " + distance / input.length);
            }
        }
    }
}