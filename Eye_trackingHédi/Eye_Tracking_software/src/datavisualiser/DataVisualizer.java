package datavisualiser;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import csvtranslater.FileManager;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.Chart;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.KeyCode;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;
import javafx.util.Pair;

/**
 * A way to review the data, requires to generate first the data and the
 * heatmaps. Shows histograms before and after the filtering.
 */
public class DataVisualizer extends Application {

    // the paths you use for each type of file
    private final static String imagesPath = "/Eye_tracking/comic_images/";
    private final static String heatmapsPath = "Z:\\bachelorProject\\resources\\outputFiles\\heatmaps\\participants\\";
    private final static String outputFilesPath = "/Eye_tracking/csvconverted/gaze/";

    // strings to create path names and labels
    private static final String AFTER_FILTERING = " After filtering";
    private static final String _PARTICIPANT = "_participant_";
    private static final String TIME_SEC = "Time (sec)";
    private static final String FIXATIONS_COUNT = "Fixations count";
    private static final String _1_BORDERLESS_PNG = "1.borderless.png";
    private static final String BEFORE_FILTERING = " Before filtering";
    private static final String CSV = ".csv";
    private static final String JPG1_BORDERLESS_PNG = ".jpg1.borderless.png";
    private static final String GAZE_POSITIONS_ON_SURFACE = "gaze_positions_on_surface_";
    private static final String FIXATIONS_ON_SURFACE = "fixations_on_surface_";
    private static final String PNG = ".png";
    private static final String SEPARATOR = ",";
    private static final String TITLE = "Data Visualiser";

    // get the all output csv files
    private final static Map<String, List<File>> inputFiles = FileManager
            .getExportsFileList();

    // list of every existing pair (participant id, image) in the dataset
    // We only store this pair in the memory instead of storing all images
    private final List<Pair<String, String>> idsNames = getIdsNamesList();

    // current page we're looking at
    private int nb = 0;

    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage stage) throws Exception {

        stage.setTitle(TITLE);
        Pair<String, String> first = idsNames.get(0);
        stage.setScene(genScene(stage, first.getKey(), first.getValue()));
        stage.show();
    }

    /**
     * Gives the list of exisitng pairs of participant ids and image name
     * 
     * @return The id and image name list
     */
    private List<Pair<String, String>> getIdsNamesList() {
        List<Pair<String, String>> idsNames = new ArrayList<>();
        for (String id : inputFiles.keySet()) {
            for (File inputFile : inputFiles.get(id)) {
                String fileName = inputFile.getName();
                String imageName = fileName
                        .replaceFirst(FIXATIONS_ON_SURFACE, "")
                        .replaceFirst(GAZE_POSITIONS_ON_SURFACE, "")
                        .replaceFirst(CSV, JPG1_BORDERLESS_PNG);

                Pair<String, String> pair = new Pair<>(id, imageName);
                idsNames.add(pair);
            }
        }
        return idsNames;
    };

    /**
     * Generates the visualization for a given image
     * 
     * @param stage
     *            the current javafx Stage
     * @param id
     *            participant id
     * @param imageName
     *            image name
     * @return the corresponding scene
     */
    private Scene genScene(Stage stage, String id, String imageName) {

        Scene scene = new Scene(getLayoutPane(id, imageName));
        scene.setOnKeyPressed((ke) -> {
            KeyCode kc = ke.getCode();
            if (kc.equals(KeyCode.RIGHT)) {
                if (nb < idsNames.size() - 1) {
                    ++nb;
                    stage.setScene(genScene(stage, idsNames.get(nb).getKey(),
                            idsNames.get(nb).getValue()));
                }
            } else if (kc.equals(KeyCode.LEFT))
                if (nb > 0) {
                    --nb;
                    stage.setScene(genScene(stage, idsNames.get(nb).getKey(),
                            idsNames.get(nb).getValue()));
                }
        });

        return scene;
    };

    private HBox getLayoutPane(String participantId, String imageName) {
        ImageView heatmap;
        try {
            FileInputStream inputstream = new FileInputStream(heatmapsPath
                    + imageName + _PARTICIPANT + participantId + PNG);
            heatmap = new ImageView(new Image(inputstream));
            inputstream.close();
        } catch (IOException e) {
            System.out.println("heatmap not found : " + heatmapsPath + imageName
                    + _PARTICIPANT + participantId + PNG);
            try {
                FileInputStream inputstream = new FileInputStream(
                        imagesPath + imageName);
                heatmap = new ImageView(new Image(inputstream));
                inputstream.close();
            } catch (IOException e2) {
                heatmap = new ImageView();
                e2.printStackTrace();
            }

        }

        return new HBox(getDualHistrogramsPane(participantId, imageName),
                heatmap);

    }

    private VBox getDualHistrogramsPane(String participantID,
            String imageName) {
        String csvInputFileName = GAZE_POSITIONS_ON_SURFACE
                + imageName.replaceFirst(JPG1_BORDERLESS_PNG, CSV);
        File inputFile = inputFileFromList(participantID, csvInputFileName);
        Chart inputHistogram = getHistogram(participantID + BEFORE_FILTERING,
                (inputFile != null ? getFrequencyListFromInput(inputFile)
                        : new HashMap<>()),
                TIME_SEC, FIXATIONS_COUNT);
        String csvOutputFileName = imageName.replaceFirst(_1_BORDERLESS_PNG,
                _PARTICIPANT + participantID + CSV);
        File outputFile = new File(outputFilesPath + csvOutputFileName);
        Chart outputHistogram = getHistogram(participantID + AFTER_FILTERING,
                (outputFile != null ? getFrequencyListFromOutput(outputFile)
                        : new HashMap<>()),
                TIME_SEC, FIXATIONS_COUNT);
        return new VBox(inputHistogram, outputHistogram);

    }

    private File inputFileFromList(String participantID, String csvFileName) {
        for (File f : inputFiles.get(participantID)) {
            if (f.getName().equals(csvFileName))
                return f;
        }
        return null;
    }

    private Chart getHistogram(String title,
            Map<Integer, Integer> pointsPerSecMap, String xAxisLabel,
            String dataLabel) {

        // creating an histogram using the javafx XYChart class
        final NumberAxis xAxis = new NumberAxis();
        final NumberAxis yAxis = new NumberAxis();
        xAxis.setLabel(xAxisLabel);
        xAxis.forceZeroInRangeProperty().set(false);
        XYChart.Series series = new XYChart.Series();
        series.setName(dataLabel);

        // fill the data series
        for (Entry<Integer, Integer> pointPerSec : pointsPerSecMap.entrySet()) {
            series.getData().add(new XYChart.Data(pointPerSec.getKey(),
                    pointPerSec.getValue()));
        }

        // creating the chart
        final LineChart<Number, Number> lineChart = new LineChart<Number, Number>(
                xAxis, yAxis);

        lineChart.setTitle(title);

        // write data on the chart
        lineChart.getData().add(series);

        return lineChart;

    }

    /**
     * Returns the list of frequencies (fixation per second) from the given
     * output csv file
     * 
     * @param path
     *            the csv file path
     * @return a list containing the frequencies
     */
    public Map<Integer, Integer> getFrequencyListFromOutput(File file) {

        Map<Integer, Integer> freqList = new HashMap<>();

        String row;

        try {
            BufferedReader csvReader = new BufferedReader(new FileReader(file));
            // skip first line
            csvReader.readLine();

            while ((row = csvReader.readLine()) != null) {
                String[] data = row.split(SEPARATOR);

                int currentSec = Integer.parseInt(data[2]) / 1000;
                if (freqList.containsKey(currentSec)) {
                    freqList.replace(currentSec, freqList.get(currentSec) + 1);
                } else {
                    freqList.put(currentSec, 1);
                }

            }
            // if (previousSec == currentSec) {
            // count++;
            // } else if (previousSec < currentSec) {
            // if (previousSec != -1) {
            // freqList.add(count);
            // }
            // count = 1;
            // previousSec = currentSec;
            // }
            // }
            // freqList.add(count);

            csvReader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return freqList;
    }

    /**
     * Returns the list of frequencies (fixation per second) from the given
     * input csv file
     * 
     * @param path
     *            the csv file path
     * @return a list containing the frequencies
     */
    public Map<Integer, Integer> getFrequencyListFromInput(File file) {

        Map<Integer, Integer> freqList = new HashMap<>();

        String row;

        try {
            BufferedReader csvReader = new BufferedReader(new FileReader(file));
            // skip first line
            csvReader.readLine();

            while ((row = csvReader.readLine()) != null) {
                String[] data = row.split(SEPARATOR);

                int currentSec = (int) Math.round(Double.parseDouble(data[2]));
                if (freqList.containsKey(currentSec)) {
                    freqList.replace(currentSec, freqList.get(currentSec) + 1);
                } else {
                    freqList.put(currentSec, 1);
                }
            }

            csvReader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return freqList;
    }

}
