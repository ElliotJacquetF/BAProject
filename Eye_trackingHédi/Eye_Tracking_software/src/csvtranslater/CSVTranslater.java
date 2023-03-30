package csvtranslater;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import javax.imageio.ImageIO;
import javax.xml.transform.stream.StreamSource;

/**
 * @class Translates the csv file from the exports of pupil player to the format
 *        that we want.
 */
public class CSVTranslater {

    private static final String X_Y = "x,y,timestamp\n";
    private static final String CSVCONVERTED = "csvconverted/";
    private static final String COMIC_IMAGES = "comic_images/";
    private static final String FIXATIONS = "fixations/";
    private static final String GAZE = "gaze/";
    private static final String CSV_EXTENTION = ".csv";
    private static final String JPG_PARTICIPANT = ".jpg_participant_";
    private static final String JPG1_BORDERLESS_PNG = ".jpg1.borderless.png";
    private final static String TEXT_GAZE_POSITIONS_ON_SURFACE = "gaze_positions_on_surface_";
    private final static String TEXT_FIXATIONS_ON_SURFACE = "fixations_on_surface_";

    private final static double DATA_REMOVE_PERCENTAGE = 0.05;
    private final static double LOW_THRESHOLD = 0.7;
    private final static double HIGH_THRESHOLD = 0.9;

    public static void main(String[] args) {

        for (Entry<String, List<File>> e : FileManager.getExportsFileList()
                .entrySet()) {
            String ID = e.getKey();
            System.out.println(FileManager.getExportsFileList());
            System.out.println(e.getKey() + " : " + e.getValue().size()
                    + " gaze/fixation files found");
            for (File f : e.getValue()) {
                String file = f.getName();
                System.out.println(file);
                if (file.startsWith(TEXT_GAZE_POSITIONS_ON_SURFACE)) {
                    String name = file
                            .replaceFirst(TEXT_GAZE_POSITIONS_ON_SURFACE, "");
                    String outputname = GAZE + name.replaceFirst(CSV_EXTENTION,
                            JPG_PARTICIPANT + ID + CSV_EXTENTION);
                    String imagename = name.replace(CSV_EXTENTION,
                            JPG1_BORDERLESS_PNG);
                    System.out.println(f);
                    System.out.println(imagename);
                    System.out.println(outputname);
                    convertCsvWithThresholdAndPercentage(f, imagename,
                            outputname, LOW_THRESHOLD, HIGH_THRESHOLD,
                            DATA_REMOVE_PERCENTAGE);
                     } /*else if (file.startsWith(TEXT_FIXATIONS_ON_SURFACE)) {
                     String name =
                     file.replaceFirst(TEXT_FIXATIONS_ON_SURFACE,
                     "");
                     String outputname = FIXATIONS
                     + name.replaceFirst(CSV_EXTENTION,
                     JPG_PARTICIPANT + ID + CSV_EXTENTION);
                     String imagename = name.replace(CSV_EXTENTION,
                     JPG1_BORDERLESS_PNG);
                     convertCsvFixationWithClustering(f, imagename,
                     outputname);
                }*/
            }
        }

        System.out.println("Success !");

    }

    /**
     * Convert the fixations from pupil player and clusters consecutive
     * fixations that are close to each other
     *
     * @param inputFile
     *            the input file
     * @param image
     *            comic page corresponding to input file
     * @param outputName
     *            output file name
     */
    private static void convertCsvFixationWithClustering(File inputFile,
                                                         String image, String outputName) {
        BufferedReader csvReader;
        try {
            BufferedImage bimg = ImageIO.read(new File(COMIC_IMAGES + image));
            int width = bimg.getWidth() - 1;
            int height = bimg.getHeight() - 1;

            csvReader = new BufferedReader(new FileReader(inputFile));
            // filter out of page data, create indexes
            String row;
            List<Set<Fixation>> fixList = new ArrayList<Set<Fixation>>();
            try {
                csvReader.readLine(); // skip first line
                int lastId = -1;
                while ((row = csvReader.readLine()) != null) {
                    String[] data = row.split(",");

                    boolean onSurface = Boolean.parseBoolean(data[10]);

                    if (onSurface) {

                        int id = Integer.parseInt(data[2]);
                        if (id > lastId) {
                            lastId = id;
                            fixList.add(new HashSet<>());
                        }

                        int timestamp = (int) Math
                                .round(Double.parseDouble(data[0]) * 1000);
                        int x = (int) Math
                                .round(Double.parseDouble(data[6]) * width);
                        int y = (int) Math.round(
                                (1 - Double.parseDouble(data[7])) * height);
                        fixList.get(fixList.size() - 1)
                                .add(new Fixation(timestamp, x, y));
                    }
                }
                csvReader.close();
            } catch (IOException e) {
                e.printStackTrace();
            }

            // create mean fixation points
            List<Fixation> meanList = new ArrayList<>();
            for (Set<Fixation> set : fixList) {
                meanList.add(Fixation.mean(set));
            }

            // remove points which are further to previous/next means than their
            // own
            assert meanList.size() == fixList.size();

            List<Set<Fixation>> definitiveFixList = new ArrayList<>();

            // make a copy of the newly created list
            for (Set<Fixation> set : fixList) {
                Set<Fixation> newSet = new HashSet<>();
                for (Fixation fixation : set) {
                    newSet.add(fixation);
                }
                definitiveFixList.add(newSet);
            }

            // clustering fixations to the nearest fixations
            for (int i = 0; i < fixList.size(); i++) {
                for (Fixation fixation : fixList.get(i)) {
                    double distToMean = fixation.dist(meanList.get(i));
                    if (i > 0 && fixation
                            .dist(meanList.get(i - 1)) < distToMean) {
                        definitiveFixList.get(i - 1).add(fixation);
                        definitiveFixList.get(i).remove(fixation);
                    } else if (i < fixList.size() - 1 && fixation
                            .dist(meanList.get(i + 1)) < distToMean) {
                        definitiveFixList.get(i + 1).add(fixation);
                        definitiveFixList.get(i).remove(fixation);
                    }
                }
            }

            List<Fixation> definitiveMeanList = new ArrayList<>();
            for (Set<Fixation> set : definitiveFixList) {
                Fixation mean = Fixation.mean(set);
                if (mean != null)
                    definitiveMeanList.add(Fixation.mean(set));
            }

            // write the results under CSVCONVERTED
            FileWriter csvWriter = new FileWriter(CSVCONVERTED + outputName);
            csvWriter.append(X_Y);

            try {
                for (Fixation f : definitiveMeanList) {
                    StringBuilder sb = new StringBuilder();
                    sb.append(f.x);
                    sb.append(',');
                    sb.append(f.y);
                    sb.append(',');
                    sb.append(f.timestamp);
                    sb.append('\n');
                    csvWriter.append(sb.toString());
                }

                csvWriter.flush();
                csvWriter.close();
            } catch (IOException e) {
                e.printStackTrace();
            }

        } catch (FileNotFoundException e1) {
            e1.printStackTrace();
        } catch (IOException e1) {
            e1.printStackTrace();
        }

    }

    /**
     * Convert the fixations from pupil player without clustering
     *
     * @param inputFile
     *            the input file
     * @param image
     *            comic page corresponding to input file
     * @param outputName
     *            output file name
     */
    private static void convertCsvFixation(File inputFile, String image,
                                           String outputName) {
        BufferedReader csvReader;
        try {
            BufferedImage bimg = ImageIO.read(new File(COMIC_IMAGES + image));
            int width = bimg.getWidth() - 1;
            int height = bimg.getHeight() - 1;

            csvReader = new BufferedReader(new FileReader(inputFile));
            String row;

            FileWriter csvWriter = new FileWriter(CSVCONVERTED + outputName);
            csvWriter.append(X_Y);

            try {
                csvReader.readLine();

                while ((row = csvReader.readLine()) != null) {
                    String[] data = row.split(",");

                    boolean onSurface = Boolean.parseBoolean(data[10]);
                    int timestamp = (int) Math
                            .round(Double.parseDouble(data[0]) * 1000);

                    // if (onSurface && confident
                    // && timestamp > previousTimeStamp) {
                    if (onSurface) {
                        // previousTimeStamp = timestamp;
                        int x = (int) Math
                                .round(Double.parseDouble(data[6]) * width);
                        int y = (int) Math.round(
                                (1 - Double.parseDouble(data[7])) * height);
                        StringBuilder sb = new StringBuilder();
                        sb.append(x);
                        sb.append(',');
                        sb.append(y);
                        sb.append(',');
                        sb.append(timestamp);
                        sb.append('\n');

                        csvWriter.append(sb.toString());
                    }
                }
                csvWriter.flush();
                csvWriter.close();
                csvReader.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        } catch (FileNotFoundException e1) {
            e1.printStackTrace();
        } catch (IOException e1) {
            e1.printStackTrace();
        }
    };

    /**
     * Convert the gaze points from pupil player to our desired CSV format by
     * only filtering out the points which are outside the page
     *
     * @param inputFile
     *            the input csv file
     * @param image
     *            comic page corresponding to input file
     * @param outputName
     *            output file name
     */
    private static void convertCsv(File inputFile, String image,
                                   String outputName) {

        BufferedReader csvReader;
        try {
            BufferedImage bimg = ImageIO.read(new File(COMIC_IMAGES + image));
            int width = bimg.getWidth() - 1;
            int height = bimg.getHeight() - 1;

            csvReader = new BufferedReader(new FileReader(inputFile));
            String row;

            FileWriter csvWriter = new FileWriter(CSVCONVERTED + outputName);
            csvWriter.append(X_Y);

            try {
                // read from input csv file
                csvReader.readLine();
                while ((row = csvReader.readLine()) != null) {

                    String[] data = row.split(",");

                    boolean onSurface = Boolean.parseBoolean(data[7]);

                    if (onSurface) {
                        // write to target output file
                        int x = (int) Math
                                .round(Double.parseDouble(data[3]) * width);
                        int y = (int) Math.round(
                                (1 - Double.parseDouble(data[4])) * height);
                        int timestamp = (int) Math
                                .round(Double.parseDouble(data[2]) * 1000);
                        StringBuilder sb = new StringBuilder();
                        sb.append(x);
                        sb.append(',');
                        sb.append(y);
                        sb.append(',');
                        sb.append(timestamp);
                        sb.append('\n');

                        csvWriter.append(sb.toString());
                    }

                }

                csvWriter.flush();
                csvWriter.close();
                csvReader.close();
            } catch (IOException e) {
                e.printStackTrace();
            }

        } catch (FileNotFoundException e1) {
            e1.printStackTrace();
        } catch (IOException e1) {
            e1.printStackTrace();
        }

    };

    /**
     * Convert the gaze points from pupil player to our desired CSV format. We
     * remove every gaze point outside the page and those whose confidence <
     * lowThreshold and keep every gaze point with confidence > highTreshold.
     * Then we remove a certain percentage of the less confident remaining data.
     *
     * @param inputFile
     *            the input csv file
     * @param image
     *            the image corresponding to input data
     * @param outputName
     *            the desired output csv file name
     * @param lowThreshold
     *            low confidence threshold
     * @param highTreshold
     *            high confidence threshold
     * @param percentage
     *            percentage of data to remove from the remaining gze points
     */
    private static void convertCsvWithThresholdAndPercentage(File inputFile,
                                                             String image, String outputName, double lowThreshold,
                                                             double highTreshold, double percentage) {
        if (lowThreshold < 0. || highTreshold > 1.
                || lowThreshold > highTreshold)
            throw new IllegalArgumentException(
                    "low and high must respect 0 <= low <= high <= 1");
        if (percentage < 0. || percentage > 1.)
            throw new IllegalArgumentException(
                    "percentage must be between 0 and 1");

        // read from input file
        BufferedReader csvReader;
        try {
            BufferedImage bimg = ImageIO.read(new File(COMIC_IMAGES + image));
            int width = bimg.getWidth() - 1;
            int height = bimg.getHeight() - 1;

            csvReader = new BufferedReader(new FileReader(inputFile));
            // read row by row
            String row;
            // final
            Map<Integer, GazePoint> tab = new HashMap<>();
            List<Pair> aList = new ArrayList<>();
            try {
                csvReader.readLine(); // skip first line
                int i = 0;
                while ((row = csvReader.readLine()) != null) {
                    String[] data = row.split(",");

                    boolean onSurface = Boolean.parseBoolean(data[7]);
                    double confidence = Double.parseDouble(data[8]);

                    if (onSurface && confidence > lowThreshold) {

                        int timestamp = (int) Math
                                .round(Double.parseDouble(data[2]) * 1000);
                        int x = (int) Math
                                .round(Double.parseDouble(data[3]) * width);
                        int y = (int) Math.round(
                                (1 - Double.parseDouble(data[4])) * height);

                        GazePoint gp = new GazePoint(timestamp, x, y,
                                confidence);
                        if (confidence < highTreshold) {
                            aList.add(new Pair(i, gp));
                        }
                        tab.put(i++, gp);

                    }
                }
                csvReader.close();
            } catch (IOException e) {
                e.printStackTrace();
            }

            Pair[] atab = new Pair[aList.size()];
            int i = 0;
            for (Pair t : aList) {
                atab[i++] = t;
            }

            Pair[] amin = getMin(atab, percentage);

            // second pass : remove less confident data
            for (Pair tuple : amin) {
                tab.remove(tuple.index);
            }

            FileWriter csvWriter = new FileWriter(CSVCONVERTED + outputName);
            csvWriter.append(X_Y);

            try {
                for (Entry<Integer, GazePoint> e : tab.entrySet()) {
                    StringBuilder sb = new StringBuilder();
                    sb.append(e.getValue().x);
                    sb.append(',');
                    sb.append(e.getValue().y);
                    sb.append(',');
                    sb.append(e.getValue().timestamp);
                    sb.append('\n');
                    csvWriter.append(sb.toString());
                }

                csvWriter.flush();
                csvWriter.close();
            } catch (IOException e) {
                e.printStackTrace();
            }

        } catch (FileNotFoundException e1) {
            e1.printStackTrace();
        } catch (IOException e1) {
            e1.printStackTrace();
        }

    }

    /**
     * Convert the gaze points from pupil player to our desired CSV format. We
     * remove every gaze point outside the page and a certain percentage of the
     * remaining data
     *
     * @param inputFile
     *            the input csv file
     * @param image
     *            the image corresponding to input data
     * @param outputName
     *            the desired output csv file name
     * @param lowThreshold
     *            low confidence threshold
     * @param highTreshold
     *            high confidence threshold
     * @param percentage
     *            percentage of data to remove from the remaining gze points
     */
    private static void convertCsvWithFilterPercentage(File inputFile,
                                                       String image, String outputname, double percentage) {
        BufferedReader csvReader;
        try {
            BufferedImage bimg = ImageIO.read(new File(COMIC_IMAGES + image));
            int width = bimg.getWidth() - 1;
            int height = bimg.getHeight() - 1;

            csvReader = new BufferedReader(new FileReader(inputFile));
            // first pass, filter out of page data, create indexes
            String row;
            Map<Integer, GazePoint> tab = new HashMap<>();
            try {
                csvReader.readLine(); // skip first line
                int i = 0;
                while ((row = csvReader.readLine()) != null) {
                    String[] data = row.split(",");

                    boolean onSurface = Boolean.parseBoolean(data[7]);

                    if (onSurface) {
                        double confidence = Double.parseDouble(data[8]);

                        int timestamp = (int) Math
                                .round(Double.parseDouble(data[2]) * 1000);
                        int x = (int) Math
                                .round(Double.parseDouble(data[3]) * width);
                        int y = (int) Math.round(
                                (1 - Double.parseDouble(data[4])) * height);
                        tab.put(i++,
                                new GazePoint(timestamp, x, y, confidence));
                    }
                }
                csvReader.close();
            } catch (IOException e) {
                e.printStackTrace();
            }

            Pair[] atab = new Pair[tab.size()];
            for (Entry<Integer, GazePoint> e : tab.entrySet()) {
                atab[e.getKey()] = new Pair(e.getKey(), e.getValue());
            }

            // less confident data
            Pair[] amin = getMin(atab, percentage);

            // remove less confident data
            for (Pair tuple : amin) {
                tab.remove(tuple.index);
            }

            FileWriter csvWriter = new FileWriter(CSVCONVERTED + outputname);
            csvWriter.append(X_Y);

            try {
                for (Entry<Integer, GazePoint> e : tab.entrySet()) {
                    StringBuilder sb = new StringBuilder();
                    sb.append(e.getValue().x);
                    sb.append(',');
                    sb.append(e.getValue().y);
                    sb.append(',');
                    sb.append(e.getValue().timestamp);
                    sb.append('\n');
                    csvWriter.append(sb.toString());
                }

                csvWriter.flush();
                csvWriter.close();
            } catch (IOException e) {
                e.printStackTrace();
            }

        } catch (FileNotFoundException e1) {
            e1.printStackTrace();
        } catch (IOException e1) {
            e1.printStackTrace();
        }

    }

    /**
     * Constructs a heap returning the Tuples which have the minimum confidence
     *
     * @param tab
     *            the array from which we take the minimum elements
     * @param percentage
     *            the percentage of data to low data confidence to return
     * @return an array containing the minimum confidence gazepoints
     */
    private static Pair[] getMin(Pair[] tab, double percentage) {
        Pair[] heap = new Pair[(int) (tab.length * percentage)];
        if (heap.length == 0) {
            return heap;
        }
        int i = 0;
        for (Pair tuple : tab) {
            if (i++ < heap.length) {
                // if not already full, build min heap
                heap[heap.length - i] = tuple;
                minHeapify(heap, heap.length - i);
            } else {
                // else try to add it to the top and min heapify
                if (heap[0].gazepoint.confidence > tuple.gazepoint.confidence) {
                    heap[0] = tuple;
                    minHeapify(heap, 0);
                }
            }
        }
        return heap;
    }

    private static void minHeapify(Pair[] heap, int index) {
        int leftindex = 2 * index + 1;
        int rightindex = leftindex + 1;

        if (rightindex < heap.length
                && heap[rightindex].gazepoint.confidence > heap[index].gazepoint.confidence) {
            if (heap[leftindex].gazepoint.confidence > heap[rightindex].gazepoint.confidence) {
                Pair temp = heap[index];
                heap[index] = heap[leftindex];
                heap[leftindex] = temp;
                minHeapify(heap, leftindex);
            } else {
                Pair temp = heap[index];
                heap[index] = heap[rightindex];
                heap[rightindex] = temp;
                minHeapify(heap, rightindex);
            }
        } else {
            if (leftindex < heap.length
                    && heap[leftindex].gazepoint.confidence > heap[index].gazepoint.confidence) {
                Pair temp = heap[index];
                heap[index] = heap[leftindex];
                heap[leftindex] = temp;
                minHeapify(heap, leftindex);
            }
        }
    }

}

class Pair {
    public final int index;
    public final GazePoint gazepoint;

    public Pair(int index, GazePoint gazepoint) {
        this.index = index;
        this.gazepoint = gazepoint;
    }

}

class GazePoint {
    public final int timestamp;
    public final int x;
    public final int y;
    public final double confidence;

    public GazePoint(int timestamp, int x, int y, double confidence) {
        this.timestamp = timestamp;
        this.x = x;
        this.y = y;
        this.confidence = confidence;
    }
}

class Fixation {

    public final int timestamp;
    public final int x;
    public final int y;

    public Fixation(int timestamp, int x, int y) {

        this.timestamp = timestamp;
        this.x = x;
        this.y = y;
    }

    /**
     * Computes the mean of a set of fixations
     *
     * @param set
     *            the set of fixation
     * @return the mean Fixation
     */
    public static Fixation mean(Set<Fixation> set) {
        if (set.size() == 0)
            return null;
        int meanX = 0;
        int meanY = 0;
        int firstTimeStamp = Integer.MAX_VALUE;

        for (Fixation fixation : set) {
            meanX += fixation.x;
            meanY += fixation.y;
            if (fixation.timestamp < firstTimeStamp)
                firstTimeStamp = fixation.timestamp;
        }
        meanX /= set.size();
        meanY /= set.size();

        return new Fixation(firstTimeStamp, meanX, meanY);
    }

    /**
     * computes the Euclidean distance between 2 fixation points
     *
     * @param that
     *            second point
     * @return euclidean distance
     */
    public double dist(Fixation that) {
        int distx = (this.x - that.x);
        int disty = (this.y - that.y);
        return Math.sqrt(distx * distx + disty * disty);
    }

}