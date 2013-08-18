package com.cts;
import com.jmatio.io.MatFileReader;
import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created with IntelliJ IDEA.
 * User: Colin Stearns
 * Date: 5/1/13
 * Time: 9:37 AM
 * To change this template use File | Settings | File Templates.
 */

/*Genres:
    1   Classical
    2   Electronic
    3   Rock/Pop
    4   World
    5   Jazz/Blues
    6   Metal/Punk
*/
public class IO {
    public static Map<String,double[][]> getSongData () {
        String dataName = "C:\\Users\\Colin Stearns\\Dropbox\\ECEN 5322 Project\\HDNeuralNet\\DeepBM\\src\\com\\cts\\songdata.mat";
        Map<String,MLArray> input = getMLArray(dataName);
        if (input == null){//if import was successful
            System.out.println("Import of data " + dataName + " failed.");
            return null;
        }

        MLDouble mlSongData = (MLDouble)input.get("songs");
        MLDouble mlGenreData = (MLDouble)input.get("genreindicies");
        double[][] songData = mlSongData.getArray();
        double[][] genreData = mlGenreData.getArray();
        double[][] songDataTranspose = new double[songData[0].length][songData.length];
        int songDimension = songData.length;
        for (int i = 0; i < songData.length; i++){
            for (int j = 0; j < songData[0].length; j++){
                songDataTranspose[j][i] = songData[i][j];
            }
        }
        normalize(songDataTranspose);
        int[] genreSizes = new int[6];
        for (int i = 0; i < genreData.length; i++){
            genreSizes[(int)genreData[i][0] - 1]++;
        }

        List<double[][]> songList = new ArrayList<double[][]>(6);
        songList.add(new double[genreSizes[0]][songDimension]);
        songList.add(new double[genreSizes[1]][songDimension]);
        songList.add(new double[genreSizes[2]][songDimension]);
        songList.add(new double[genreSizes[3]][songDimension]);
        songList.add(new double[genreSizes[4]][songDimension]);
        songList.add(new double[genreSizes[5]][songDimension]);
        int[] genreIndices = {0,0,0,0,0,0};

        for (int i = 0; i < genreData.length; i++){
            int genre = (int)genreData[i][0] - 1;
            songList.get(genre)[genreIndices[genre]] = songDataTranspose[i];
            genreIndices[genre]++;
        }

        Map<String,double[][]> songs = new HashMap<String,double[][]>();
        songs.put("classical", songList.get(0));
        songs.put("electronics", songList.get(1));
        songs.put("rock_pop", songList.get(2));
        songs.put("world", songList.get(3));
        songs.put("jazz_blues", songList.get(4));
        songs.put("metal_punk", songList.get(5));
        songs.put("allSongs",songDataTranspose);
        return songs;
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

    private static void normalize (double[][] data){
        int numPoints = data.length;
        int dimension = data[0].length;
        double[] min = new double[dimension];
        double[] max = new double[dimension];

        //initialize to values of first point
        for (int j = 0; j < dimension; j++){
            min[j] = data[0][j];
            max[j] = data[0][j];
        }

        for (int i = 1; i < numPoints; i++){
            for (int j = 0; j < dimension; j++){
                min[j] = Math.min(min[j],data[i][j]);
                max[j] = Math.max(max[j],data[i][j]);
            }
        }

        for (int i = 0; i < numPoints; i++){
            for (int j = 0; j < dimension; j++){
                double newDatum = data[i][j];
                newDatum -= min[j];//normalize to zero
                newDatum = newDatum/(max[j] - min[j]);//divide by normalized maximum
                data[i][j] = newDatum;
            }
        }

    }
}
