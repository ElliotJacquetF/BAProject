package csvtranslater;

import javax.sound.midi.Soundbank;
import java.io.File;
import java.sql.SQLOutput;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author Florian
 * @class A class containing static methods to get export files
 */
public class FileManager {

    // Recordings location in the file manager
    private final static String RECORDINGS_LOCATION = "/home/hidri/Documents/recordings/";

    //
    private static final int GAZE_FILE_EMPTY_SIZE = 95;
    private static final int FIXATION_FILE_EMPTY_SIZE = 125;
    private static final String GAZE_POSITIONS_ON_SURFACE = "gaze_positions_on_surface_";
    private static final String FIXATIONS_ON_SURFACE = "fixations_on_surface_";
    private static final String SURFACES = "/surfaces";
    private static final String EXPORTS = "/exports";

    // not instanciable
    private FileManager() {
    }

    /**
     * Gets the most recent exports csv fixation and gaze points files for each
     * participant
     * 
     * @return a map with the participant IDs as Key and the list of
     *         corresponding csv files as Values
     */
    public static Map<String, List<File>> getExportsFileList() {
        File directory = new File(RECORDINGS_LOCATION);
        Map<String, List<File>> exportsMap = new HashMap<String, List<File>>();
        //System.out.println(directory);
        for (File f : directory.listFiles()) {
            //System.out.println(f);
            if (f.isDirectory()) {

                exportsMap.put(f.getName(), getMostRecentExports(f));
            }
        }

        return exportsMap;

    }

    /**
     * Finds the most recent exports (the one whose name is the biggest integer)
     * 
     * @param directory
     *            the recordings directory
     * @return the list of every most recent exports files for each surface
     *         (AOI)
     */
    private static List<File> getMostRecentExports(File directory) {

        List<File> filesList = new ArrayList<File>();
        for (File subDir : directory.listFiles()) {
            File exportsDir = new File(subDir.getAbsolutePath() + EXPORTS);
            //System.out.println(exportsDir);
            System.out.println(subDir.getAbsolutePath() + EXPORTS);
            //System.out.println(directory.listFiles());
            //System.out.println(exportsDir.getAbsoluteFile().exists());
            //System.out.println(exportsDir.list().length);
            if (exportsDir.exists() && exportsDir.list().length > 0) {

                File mostRecent = null;
                for (File csvDir : exportsDir.listFiles()) {
                    if (mostRecent == null
                            || Long.parseLong(mostRecent.getName()) < Long
                                    .parseLong(csvDir.getName())) {
                        mostRecent = csvDir;
                    }
                }
                filesList.addAll(addAllIfNotEmpty(mostRecent));
            }
        }

        return filesList;

    }

    /**
     * Checks that there is some data for each AOI and puts it in the list if
     * there is some
     * 
     * @param file
     *            an export file from pupil player
     * @return The list of every export in the given export directory
     */
    private static List<File> addAllIfNotEmpty(File file) {
        List<File> filesList = new ArrayList<>();
        File surf = new File(file.getAbsolutePath() + SURFACES);
        if (surf.exists()) {
            for (File f : surf.listFiles()) {
                if ((f.getName().startsWith(FIXATIONS_ON_SURFACE)
                        && f.length() > FIXATION_FILE_EMPTY_SIZE)
                        || (f.getName().startsWith(GAZE_POSITIONS_ON_SURFACE)
                                && f.length() > GAZE_FILE_EMPTY_SIZE)) {
                    filesList.add(f);
                }
            }
        }
        return filesList;
    }
}
