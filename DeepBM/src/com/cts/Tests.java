package com.cts;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;

/**
 * Created with IntelliJ IDEA.
 * User: Colin Stearns
 * Date: 4/20/13
 * Time: 11:30 AM
 * To change this template use File | Settings | File Templates.
 */
public class Tests {

    public static void test1(){
        RbmLayer input = new RbmLayer(5);
        RbmLayer testLayer = new RbmLayer (10);
        System.out.println("Test Biases: ");
        System.out.println(Arrays.toString(testLayer.getBiases()));
        System.out.println();
        double[] testInputs = {1,1,1,1,1};
        System.out.printf("\nInput layer outputs for input {1,1,1,1,1}:\n");
        System.out.println(Arrays.toString(testLayer.getOutputs(testInputs)));
        System.out.printf("\nTest layer outputs:\n");
        System.out.println(Arrays.toString(testLayer.getOutputs(testInputs)));

        double[] test2Inputs = {0.5, 0.5, 0.5, 0.5, 0.5};
        System.out.printf("\nInput layer outputs for input {0.5, 0.5, 0.5, 0.5, 0.5}:\n");
        System.out.println(Arrays.toString(testLayer.getOutputs(test2Inputs)));
        System.out.printf("\nTest layer outputs:\n");
        System.out.println(Arrays.toString(testLayer.getOutputs(test2Inputs)));

        double[] test3Inputs = {0.0, 0.0, 0.0, 0.0, 0.0};
        System.out.printf("\nInput layer outputs for input {0.0, 0.0, 0.0, 0.0, 0.0}:\n");
        System.out.println(Arrays.toString(testLayer.getOutputs(test3Inputs)));
        System.out.printf("\nTest layer outputs:\n");
        System.out.println(Arrays.toString(testLayer.getOutputs(test3Inputs)));

    }

    public static void test2() {
        int[] layerSizes = {6,4,3,2,1};
        Dbm testNet = new Dbm(layerSizes);
        double[] input = {0.68, 0.23, 0.1, 0.0, 0.36, 1};
        System.out.println(Arrays.toString(testNet.getOutput(input)));
        System.out.println(Arrays.toString(testNet.getOutput(input)));
        System.out.println(Arrays.toString(testNet.getOutput(input)));
        System.out.println(Arrays.toString(testNet.getOutput(input)));
        System.out.println(Arrays.toString(testNet.getOutput(input)));
        System.out.println();

        int[] layerSizes2 = {6,500,400,300,100,30};
        Dbm testNet2 = new Dbm(layerSizes2);

        System.out.println(Arrays.toString(testNet2.getOutput(input)));
        System.out.println(Arrays.toString(testNet2.getOutput(input)));
        System.out.println(Arrays.toString(testNet2.getOutput(input)));
        System.out.println(Arrays.toString(testNet2.getOutput(input)));
        System.out.println(Arrays.toString(testNet2.getOutput(input)));

    }

    public static void test3() {
        int[] layerSizes = {2,2};
        double[] input = {0.5,0.5};
        Dbm testNet = new Dbm(layerSizes);

        System.out.println(Arrays.toString(testNet.getOutput(input)));

        List<RbmLayer> layers = testNet.getLayers();
        int i = 0;
        for (RbmLayer layer : layers){
            System.out.println("layer " + i);
            System.out.println(Arrays.toString(layer.getState()));
            System.out.println(Arrays.toString(layer.getBiases()));
            System.out.println(Arrays.toString(layer.getOutputs(input)));
            System.out.println("layer size: " + layer.getLayerSize());
            i++;
        }
    }

    public static void test4() {
        double start;
        Random rand = new Random();
        double[][] trainingData = new double[100][10];
        for (int i = 0; i < trainingData.length; i++){
            start = 0.9*i/100;
            for (int j = 0; j < 10; j++){
                trainingData[i][j] = 0.65;
            }
        }
        int[] layerSizes = {10,50,100,50,10};
        printNetData(layerSizes,trainingData);
    }

    public static void test5() {
        Map<String, double[][]> data = IO.getSongData();
        for (String key : data.keySet()){
            System.out.println(key + ", size: " + data.get(key).length);
        }
    }

    public static void test6(){
        Map<String, double[][]> data = IO.getSongData();
        double[][] trainingData = data.get("classical");
        int dimension = trainingData[0].length;
        int[] layerSizes = {dimension, dimension,dimension,dimension};
        Dbm testNet = new Dbm(layerSizes);
        testNet.train(trainingData);
    }

    public static void test7(){
        Map<String, double[][]> data = IO.getSongData();
        double[][] classical = data.get("classical");
        for (int i = 0; i < classical.length; i++){
            System.out.println(Arrays.toString(classical[i]));
        }
    }

    public static void test8(){
        double[][] trainingData = new double[40][50];
        for (int i = 0; i < trainingData.length; i++){
            for (int j = 0; j < trainingData[0].length; j++){
                trainingData[i][j] = 0.73;
            }
        }
        int dimension = trainingData[0].length;
        int[] layerSizes = {dimension, dimension};
        Dbm testNet = new Dbm(layerSizes);
        testNet.train(trainingData);
    }

    public static void test9(){
        double[] a = {1.0,0};
        double[] b = {0,1};
        double c = Analysis.getDistance(a,b);
        System.out.println(c);
        System.out.println(Math.pow(c,2));
    }










    public static void ioTest1 () {
        String dataName = "C:\\Users\\Colin Stearns\\Dropbox\\ECEN 5322 Project\\HDNeuralNet\\DeepBM\\src\\com\\cts\\songdata.mat";
        Map<String,MLArray> input = getMLArray(dataName);
        if (input == null){//if import was successful
            System.out.println("Import of data " + dataName + " failed.");
            return;
        }

        MLDouble mlSongData = (MLDouble)input.get("songs");
        MLDouble mlGenreData = (MLDouble)input.get("genreindicies");
        double[][] songData = mlSongData.getArray();
        double[][] genreData = mlGenreData.getArray();
        System.out.println("Song data dimensions: " + songData.length + " x " + songData[0].length);
        System.out.println("Genre data dimensions: " + genreData.length + " x " + genreData[0].length);
        System.out.println(songData[0][0]);
        System.out.println(genreData[0][0]);


    }

    private static Map<String,MLArray> getMLArray (String fileName){
        MatFileReader input;
        try {
            input = new MatFileReader(fileName);
            return input.getContent();
        } catch (IOException e){
            System.out.println(e.toString());
            return null;
        }
    }

    private static void printNetData (int[] layerSizes, double[][] trainingData){
        Dbm testNet = new Dbm(layerSizes);
        for (int i = 0; i < trainingData.length; i++){
            System.out.println("Untrained output: " + i);
            System.out.println("Input: " + Arrays.toString(trainingData[i]));
            System.out.println("Output: " + Arrays.toString(testNet.getOutput(trainingData[i])));
        }

        System.out.println();
        for (int i = 0; i < trainingData.length; i++){
            System.out.println("Untrained model: " + i);
            System.out.println("Input: " + Arrays.toString(trainingData[i]));
            System.out.println("Model: " + Arrays.toString(testNet.getModel(trainingData[i])));
        }

        testNet.train(trainingData);

        System.out.println();
        for (int i = 0; i < trainingData.length; i++){
            System.out.println("Trained output: " + i);
            System.out.println("Input: " + Arrays.toString(trainingData[i]));
            System.out.println("Output: " + Arrays.toString(testNet.getOutput(trainingData[i])));
        }

        System.out.println();
        for (int i = 0; i < trainingData.length; i++){
            System.out.println("Trained model: " + i);
            System.out.println("Input: " + Arrays.toString(trainingData[i]));
            System.out.println("Model: " + Arrays.toString(testNet.getModel(trainingData[i])));
        }
    }



}
